import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.special import gamma
from scipy.optimize import differential_evolution, minimize
from concurrent.futures import ThreadPoolExecutor
import warnings
import seaborn as sns
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy.stats import norm
from scipy.special import logsumexp
import statsmodels.tools.numdiff as nd
from statsmodels.tools.eval_measures import aic, bic, hqic
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import mstats, norm
from scipy.special import gamma
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit, KFold
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
import os
from scipy.stats import norm
from scipy.special import logsumexp
import json
from numpy.linalg import inv
from scipy.stats import t, kendalltau, spearmanr

from statsmodels.regression.quantile_regression import QuantReg
# 定义文件路径字典

# 修改为您的文件路径
data_files = {
    'BTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/BTCUSDT_5m.csv",
    'DASH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/DASHUSDT_5m.csv",
    'ETH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/ETHUSDT_5m.csv",
    'LTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/LTCUSDT_5m.csv",
    'XLM': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/XLMUSDT_5m.csv",
    'XRP': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/XRPUSDT_5m.csv"
}

# 定义计算收益率的函数
def calculate_returns(prices):
    # 计算日收益率：(P_t / P_{t-1} - 1) * 100
    returns = (prices / prices.shift(1) - 1) * 100
    returns.iloc[0] = 0  # 第一个收益率设为0
    returns[prices.shift(1) == 0] = np.nan  # 处理除零情况
    return returns


# 定义计算RV的函数
def calculate_rv(df, coin_name):
    # 复制数据并重命名列
    data_ret = df[['time', 'close']].copy()
    data_ret.columns = ['DT', 'PRICE']
    data_ret = data_ret.dropna()  # 删除缺失值

    # 计算收益率
    data_ret['Ret'] = calculate_returns(data_ret['PRICE'])

    # 将DT转换为日期
    data_ret['DT'] = pd.to_datetime(data_ret['DT']).dt.date

    # 计算日度RV：收益率平方的日度总和
    RV = (data_ret
          .groupby('DT')['Ret']
          .apply(lambda x: np.sum(x ** 2))
          .reset_index())

    # 重命名RV列为币种名称
    RV.columns = ['DT', f'RV_{coin_name}']
    return RV


# 计算每种加密货币的RV
rv_dfs = []
for coin, file_path in data_files.items():
    df = pd.read_csv(file_path)
    rv_df = calculate_rv(df, coin)
    rv_dfs.append(rv_df)

# 合并所有RV数据框，按DT对齐
rv_merged = rv_dfs[0]  # 以第一个RV数据框（BTC）为基础
for rv_df in rv_dfs[1:]:
    rv_merged = rv_merged.merge(rv_df, on='DT', how='outer')

# 将DT转换为datetime格式（可选）
rv_merged['DT'] = pd.to_datetime(rv_merged['DT'])

# 按日期排序
all_RV = rv_merged.sort_values('DT').reset_index(drop=True)

all_RV= all_RV.dropna()  # 删除包含NaN的行

# 数据准备
all_RV.columns = ["DT","BTC", "DASH","ETH","LTC","XLM","XRP"]

features = all_RV.drop(columns=['DT'])
data = features.to_numpy()

columns = ["BTC", "DASH","ETH","LTC","XLM","XRP"]
data_df = pd.DataFrame(data, columns=columns)

import pandas as pd
import numpy as np

# 确保 all_RV 的 DT 列是 datetime 格式
all_RV['DT'] = pd.to_datetime(all_RV['DT'])

# 步骤1：创建 model_data，以 BTC RV 作为目标变量
model_data = all_RV[['DT', 'BTC', 'DASH', 'ETH', 'LTC', 'XLM', 'XRP']].copy()
model_data = model_data.rename(columns={'BTC': 'RV'})  # 重命名 BTC 为 RV

# 步骤2：计算 BTC RV 的分位数
rv_quantiles = model_data['RV'].quantile([0.05, 0.95])
print(f"τ=0.05 分位数: {rv_quantiles[0.05]:.6f}")
print(f"τ=0.95 分位数: {rv_quantiles[0.95]:.6f}")

# 步骤3：定义市场状态
model_data['market_state'] = pd.cut(model_data['RV'],
                                    bins=[-np.inf, rv_quantiles[0.05], rv_quantiles[0.95], np.inf],
                                    labels=['Bear', 'Normal', 'Bull'])

# 步骤4：根据市场状态动态选择 RV（滞后一期）
model_data['Dynamic_RV_lag1'] = np.where(model_data['market_state'] == 'Bear',
                                         model_data['XLM'].shift(1),  # 熊市用 XLM
                                         np.where(model_data['market_state'] == 'Bull',
                                                  model_data['XRP'].shift(1),  # 牛市用 XRP
                                                  model_data['XLM'].shift(1)))  # 正常用 ETH

# 步骤5：检查生成的驱动变量
print("\n分位数状态指标前几行：")
print(model_data[['DT', 'RV', 'market_state', 'Dynamic_RV_lag1']].head(10))

# 创建滞后变量
rv_lag1 = data_df['BTC'].shift(1)
rv_lag5 = data_df['BTC'].rolling(window=5).mean().shift(1)
rv_lag22 = data_df['BTC'].rolling(window=22).mean().shift(1)
btc_lag1 = model_data['Dynamic_RV_lag1']


model1 = pd.DataFrame({
    'RV': data_df['BTC'],
    'rv_lag1': rv_lag1,
    'rv_lag5': rv_lag5,
    'rv_lag22': rv_lag22,
    'BTC_lag1': btc_lag1
}).dropna()

model_data = model1  # 使用新的数据结构

window_size = 1800
test_size = 300

train_data_full = model_data.iloc[:len(model_data) - test_size]
train_data = train_data_full.iloc[-window_size:]
test_data = model_data.iloc[len(model_data) - test_size:].reset_index(drop=True)

import pandas as pd
import numpy as np
import os
from scipy.stats import norm
from scipy.special import logsumexp
from numpy.linalg import inv
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


# --- 3. 马尔可夫切换模型核心函数 ---
def forward_backward(y, X, betas, sigmas, P, pi):
    """前向后向算法计算平滑概率"""
    T = len(y)
    K = len(sigmas)

    # E-step part 1: Likelihood of observations
    log_lik_k = np.zeros((T, K))
    for k in range(K):
        residuals = y - X @ betas[k]
        log_lik_k[:, k] = norm.logpdf(residuals, scale=np.maximum(sigmas[k], 1e-9))

    # E-step part 2: Forward pass (alpha)
    log_alpha = np.zeros((T, K))
    log_P = np.log(P + 1e-12)
    log_alpha[0] = np.log(pi + 1e-12) + log_lik_k[0]

    for t in range(1, T):
        log_alpha[t, :] = log_lik_k[t] + logsumexp(log_alpha[t - 1, :, np.newaxis] + log_P, axis=0)

    # E-step part 3: Backward pass (beta)
    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        log_beta[t, :] = logsumexp(log_P + log_lik_k[t + 1, :] + log_beta[t + 1, :], axis=1)

    # E-step part 4: Smoothed probabilities (gamma)
    log_gamma = log_alpha + log_beta
    gamma = np.exp(log_gamma - logsumexp(log_gamma, axis=1, keepdims=True))

    log_likelihood = logsumexp(log_alpha[-1])

    return gamma, log_likelihood, log_alpha, log_beta, log_lik_k


def em_mshar(model_data, max_iter=100, tol=1e-6):
    """EM算法估计MS-HAR模型参数"""
    X, y = prepare_X_y(model_data)
    T, p = X.shape
    K = 2

    # 初始化参数
    ols_beta = np.linalg.solve(X.T @ X + 1e-6 * np.eye(p), X.T @ y)
    betas = np.array([ols_beta * 0.9, ols_beta * 1.1])
    ols_res = y - X @ ols_beta
    std_resid = np.std(ols_res)
    sigmas = np.array([0.8 * std_resid, 1.2 * std_resid])
    P = np.array([[0.95, 0.05], [0.05, 0.95]])
    pi = np.array([0.5, 0.5])

    old_log_likelihood = -np.inf

    for iteration in range(max_iter):
        # E-步
        gamma, log_likelihood, log_alpha, log_beta, log_lik_k = forward_backward(y, X, betas, sigmas, P, pi)

        if not np.isfinite(log_likelihood) or abs(log_likelihood - old_log_likelihood) < tol:
            break
        old_log_likelihood = log_likelihood

        # M-步: 更新参数
        for k in range(K):
            W = np.diag(gamma[:, k] + 1e-9)
            ridge = 1e-6 * np.eye(p)
            betas[k] = np.linalg.solve(X.T @ W @ X + ridge, X.T @ W @ y)
            residuals = y - X @ betas[k]
            sigmas[k] = np.sqrt(np.sum(gamma[:, k] * residuals ** 2) / np.sum(gamma[:, k]))
            sigmas[k] = np.maximum(sigmas[k], 1e-9)

        log_P = np.log(P + 1e-12)
        log_xi_un = (log_alpha[:-1, :, np.newaxis] + log_P +
                     log_lik_k[1:, np.newaxis, :] + log_beta[1:, np.newaxis, :])
        log_xi_sum = logsumexp(log_xi_un, axis=(1, 2), keepdims=True)
        log_xi = log_xi_un - log_xi_sum
        xi = np.exp(log_xi)

        P = np.sum(xi, axis=0)
        P /= np.sum(P, axis=1, keepdims=True)

        pi = gamma[0] / np.sum(gamma[0])

    return betas, sigmas, P, pi, gamma


def prepare_X_y(model_data):
    """从DataFrame中准备X和y - 修改为包含BTC_lag1特征"""
    X = np.column_stack([
        np.ones(len(model_data)),  # 截距项
        model_data['rv_lag1'].values,  # RV滞后1期
        model_data['rv_lag5'].values,  # RV滞后5期均值
        model_data['rv_lag22'].values,  # RV滞后22期均值
        model_data['BTC_lag1'].values  # BTC滞后1期 (新增特征)
    ])
    y = model_data['RV'].values
    return X, y


# --- 4. 滚窗预测核心逻辑 ---
def rolling_window_mshar_prediction_1_step(initial_train_data, test_data):
    """
    使用滚动窗口进行1步预测，返回预测值和实际值列表。
    如果预测值为负，则用当前窗口均值替代。
    """
    predictions = []
    actuals = []

    rolling_data = initial_train_data.copy()

    for i in tqdm(range(len(test_data)), desc="Rolling MS-HAR Forecast"):
        try:
            # 计算当前窗口的均值，以备不时之需
            current_window_mean = rolling_data['RV'].mean()

            # 1. 在当前窗口上拟合模型
            betas, sigmas, P, pi, gamma = em_mshar(rolling_data)

            # 2. 准备下一步的特征
            X_1step, _ = prepare_X_y(test_data.iloc[i:i + 1])

            # 3. 计算预测值
            regime_predictions = X_1step @ betas.T
            last_smoothed_prob = gamma[-1]
            pred_prob_1step = P.T @ last_smoothed_prob
            pred_1 = np.sum(regime_predictions * pred_prob_1step)

            # 检查预测值是否为负，如果是，则进行替换
            if pred_1 < 0:
                final_pred = current_window_mean
            else:
                final_pred = pred_1

            predictions.append(final_pred)
            actuals.append(test_data['RV'].iloc[i])

        except Exception as e:
            # 如果某一步出错，记录为NaN并继续
            print(f"\n预测点 {i} 发生错误: {str(e)}. 记录为NaN.")
            predictions.append(np.nan)
            actuals.append(test_data['RV'].iloc[i])

        # 4. 更新滚动窗口: 去掉最旧的数据，加上最新的真实数据
        new_point = test_data.iloc[i:i + 1].copy()
        rolling_data = pd.concat([rolling_data.iloc[1:], new_point], ignore_index=True)

    return predictions, actuals


# --- 5. 主程序入口 ---
if __name__ == "__main__":
    # 执行预测
    predictions_1step, actuals_1step = rolling_window_mshar_prediction_1_step(
        initial_train_data=train_data,
        test_data=test_data
    )

    print("预测完成。")

    # 创建最终的结果DataFrame
    results_df = pd.DataFrame({
        'Actual_RV': actuals_1step,
        'Predicted_RV': predictions_1step
    }).dropna()  # 丢掉预测失败的行

    predictions = results_df['Predicted_RV'].values
    actuals = results_df['Actual_RV'].values

    # 计算评估指标
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    hmse = np.mean((1 - predictions / actuals) ** 2)
    hmae = np.mean(np.abs(1 - predictions / actuals))
    qlike = np.mean(np.log(predictions) + actuals / predictions)

    # 打印结果
    print(f"1-Step Prediction Loss Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"HMSE: {hmse:.6f}")
    print(f"HMAE: {hmae:.6f}")
    print(f"QLIKE: {qlike:.6f}")

    # 保存到CSV文件
    output_filename = 'mshar_predictions_with_btc_lag1.csv'
    results_df.to_csv(output_filename, index=False)

    print(f"\n预测结果已保存到文件: {output_filename}")
    print("\n结果预览:")
    print(results_df.head())