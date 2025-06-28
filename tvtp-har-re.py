import pandas as pd
import numpy as np
from scipy.stats import norm
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
# 定义数据文件路径

data_files = {
        'BTC': "BTCUSDT_5m.csv",
        'DASH': "DASHUSDT_5m.csv",
        'ETH': "ETHUSDT_5m.csv",
        'LTC': "LTCUSDT_5m.csv",
        'XLM': "XLMUSDT_5m.csv",
        'XRP': "XRPUSDT_5m.csv"
}
def get_re(data, alpha):
    """
    计算REX指标的函数
    """
    # 将数据转换为DataFrame并确保是副本
    result = data.copy()

    # 转换时间列 - 使用更健壮的方式处理日期
    try:
        # 如果时间格式是 "YYYY/M/D H" 这种格式
        result['day'] = pd.to_datetime(result['time'], format='%Y/%m/%d %H')
    except:
        try:
            # 如果上面的格式不工作，尝试其他常见格式
            result['day'] = pd.to_datetime(result['time'])
        except:
            # 如果还是不行，尝试先分割时间字符串
            result['day'] = pd.to_datetime(result['time'].str.split().str[0])

    # 只保留日期部分
    result['day'] = result['day'].dt.date

    # 按天分组进行计算
    def calculate_daily_metrics(group):
        # 计算简单收益率
        group['Ret'] = (group['close'] / group['close'].shift(1) - 1) * 100
        group.loc[group.index[0], 'Ret'] = 0  # First return is 0
        group.loc[group['close'].shift(1) == 0, 'Ret'] = np.nan  # Handle division by zero

        # 删除缺失值
        group = group.dropna()

        if len(group) == 0:
            return None

        # 计算标准差
        sigma = group['Ret'].std()

        # 计算分位数阈值
        r_minus = norm.ppf(alpha) * sigma
        r_plus = norm.ppf(1 - alpha) * sigma

        # 计算超额收益
        REX_minus = np.sum(np.where(group['Ret'] <= r_minus, group['Ret'] ** 2, 0))
        REX_plus = np.sum(np.where(group['Ret'] >= r_plus, group['Ret'] ** 2, 0))
        REX_moderate = np.sum(np.where((group['Ret'] > r_minus) & (group['Ret'] < r_plus),
                                       group['Ret'] ** 2, 0))

        return pd.Series({
            'REX_minus': REX_minus,
            'REX_plus': REX_plus,
            'REX_moderate': REX_moderate
        })

    # 按天分组计算指标
    result = result.groupby('day', group_keys=False).apply(calculate_daily_metrics, include_groups=False).reset_index()
    # 过滤掉None值
    result = result.dropna()

    return result


# 存储所有加密货币的REX数据
all_rex_data = {}

# 处理每个加密货币文件
for crypto_name, file_path in data_files.items():
    print(f"正在处理 {crypto_name}...")

    try:
        # 读取数据
        df = pd.read_csv(file_path)

        # 如果文件中有code列，过滤对应的数据；否则直接使用所有数据
        if 'code' in df.columns:
            data_filtered = df[df['code'] == crypto_name].copy()
        else:
            data_filtered = df.copy()

        # 计算REX指标
        har_re = get_re(data_filtered, alpha=0.05)

        # 添加加密货币标识列
        har_re['crypto'] = crypto_name

        # 存储结果
        all_rex_data[crypto_name] = har_re

        print(f"{crypto_name} 处理完成，共 {len(har_re)} 天数据")

    except Exception as e:
        print(f"处理 {crypto_name} 时出错: {e}")

# 合并所有数据
if all_rex_data:
    # 将所有数据合并成一个DataFrame
    combined_data = pd.concat(all_rex_data.values(), ignore_index=True)

    # 创建三个分别的数据集
    # 1. REX_minus 数据 (包含day, crypto, REX_minus)
    rex_minus_data = combined_data[['day', 'crypto', 'REX_minus']].copy()
    rex_minus_pivot = rex_minus_data.pivot(index='day', columns='crypto', values='REX_minus')
    rex_minus_pivot.index.name = 'DT'
    all_RD = rex_minus_pivot

    # 2. REX_plus 数据 (包含day, crypto, REX_plus)
    rex_plus_data = combined_data[['day', 'crypto', 'REX_plus']].copy()
    rex_plus_pivot = rex_plus_data.pivot(index='day', columns='crypto', values='REX_plus')
    rex_plus_pivot.index.name = 'DT'
    all_RP = rex_plus_pivot


    # 3. REX_moderate 数据 (包含day, crypto, REX_moderate)
    rex_moderate_data = combined_data[['day', 'crypto', 'REX_moderate']].copy()
    rex_moderate_pivot = rex_moderate_data.pivot(index='day', columns='crypto', values='REX_moderate')
    rex_moderate_pivot.index.name = 'DT'
    all_RM = rex_moderate_pivot

print(all_RM)
print(all_RD)
print(all_RP)
# ... (您上面所有的代码，直到生成了 all_RD, all_RP, all_RM)

# 确保索引是datetime格式，以便后续合并
all_RD.index = pd.to_datetime(all_RD.index)
all_RP.index = pd.to_datetime(all_RP.index)
all_RM.index = pd.to_datetime(all_RM.index)

# 复制数据以进行操作
data_rd = all_RD.copy()

# 1a. 定义市场状态：基于 BTC 的 REX_minus
rd_quantiles = data_rd['BTC'].quantile([0.05, 0.95])
data_rd['market_state'] = pd.cut(data_rd['BTC'],
                                   bins=[-np.inf, rd_quantiles[0.05], rd_quantiles[0.95], np.inf],
                                   labels=['Calm', 'Normal', 'Panic'])

# 1b. 构建动态驱动变量 Dynamic_RD_lag1
data_rd['Dynamic_RD_lag1'] = np.where(
    data_rd['market_state'] == 'Panic',
    data_rd['XRP'].shift(1),  # 恐慌状态，使用滞后一期的 XRP 的 REX_minus
    data_rd['XLM'].shift(1)   # 平静/正常状态，使用滞后一期的 XLM 的 REX_minus
)

# 1c. 整理并展示结果
final_output_rd = data_rd[['BTC', 'market_state', 'XLM', 'XRP', 'Dynamic_RD_lag1']].dropna()
data_rp = all_RP.copy()
# 2a. 定义市场状态：基于 BTC 的 REX_plus
rp_quantiles = data_rp['BTC'].quantile([0.05, 0.95])
data_rp['market_state'] = pd.cut(data_rp['BTC'],
                                    bins=[-np.inf, rp_quantiles[0.05], rp_quantiles[0.95], np.inf],
                                    labels=['Bear', 'Normal', 'Bull'])

# 2b. 构建动态驱动变量 Dynamic_RP_lag1
data_rp['Dynamic_RP_lag1'] = np.where(
    data_rp['market_state'] == 'Bull',
    data_rp['XRP'].shift(1),  # 牛市状态，使用滞后一期的 XRP 的 REX_plus
    data_rp['XLM'].shift(1)   # 熊市/正常状态，使用滞后一期的 XLM 的 REX_plus
)

# 2c. 整理并展示结果
final_output_rp = data_rp[['BTC', 'market_state', 'XLM', 'XRP', 'Dynamic_RP_lag1']].dropna()


# 复制数据
data_rm = all_RM.copy()

# 3a. 定义市场状态：基于 BTC 的 REX_moderate
rm_quantiles = data_rm['BTC'].quantile([0.05, 0.95])
data_rm['market_state'] = pd.cut(data_rm['BTC'],
                                    bins=[-np.inf, rm_quantiles[0.05], rm_quantiles[0.95], np.inf],
                                    labels=['Low_Mod', 'Normal_Mod', 'High_Mod'])

# 3b. 构建动态驱动变量 Dynamic_RM_lag1
data_rm['Dynamic_RM_lag1'] = np.where(
    data_rm['market_state'] == 'High_Mod',
    data_rm['XRP'].shift(1),  # 高活跃度状态，使用滞后一期的 XRP 的 REX_moderate
    data_rm['LTC'].shift(1)   # 低/正常活跃度状态，使用滞后一期的 LTC 的 REX_moderate
)

# 3c. 整理并展示结果
final_output_rm = data_rm[['BTC', 'market_state', 'LTC', 'XRP', 'Dynamic_RM_lag1']].dropna()



# Read the data
df_data = pd.read_csv("BTCUSDT_5m.csv")

# Get group summary
group_summary = df_data.groupby('code').size().reset_index(name='NumObservations')

# Create data_ret DataFrame with renamed columns first
data_ret = df_data[['time', 'code', 'close']].copy()
data_ret.columns = ['DT', 'id', 'PRICE']
data_ret = data_ret.dropna()

# Calculate returns for each group using the new formula
def calculate_returns(prices):
    # Compute daily returns using the given formula
    returns = (prices / prices.shift(1) - 1)*100
    returns.iloc[0] = 0  # First return is 0
    returns[prices.shift(1) == 0] = np.nan  # Handle division by zero
    return returns

# Calculate returns by group
data_ret['Ret'] = data_ret.groupby('id')['PRICE'].transform(calculate_returns)

# Get group summary for data_ret
group_summary_ret = data_ret.groupby('id').size().reset_index(name='NumObservations')

# Filter for "000001.XSHG" and remove unnecessary columns
data_filtered = data_ret[data_ret['id'] == "BTC"].copy()
data_filtered = data_filtered.drop('id', axis=1)
from datetime import date
# Convert DT to datetime and calculate daily RV
data_filtered['DT'] = pd.to_datetime(data_filtered['DT']).dt.date
# Filter using date objects
data_filtered = data_filtered[
    (data_filtered['DT'] >= date(2019, 3, 28)) &
    (data_filtered['DT'] <= date(2025, 3, 30))
]
RV_series = (data_filtered
      .groupby('DT')['Ret']
      .apply(lambda x: np.sum(x**2)))


RV = RV_series.to_frame(name='RV')

RV.index = pd.to_datetime(RV.index)
RV.index.name = 'DT'
rex_m_lag1 = all_RM['BTC'].shift(1)
rex_m_lag5 =all_RM['BTC'].rolling(window=5).mean().shift(1)
rex_m_lag22 = all_RM['BTC'].rolling(window=22).mean().shift(1)
rex_p_lag1 = all_RP['BTC'].shift(1)
rex_p_lag5 = all_RP['BTC'].rolling(window=5).mean().shift(1)
rex_p_lag22 = all_RP['BTC'].rolling(window=22).mean().shift(1)
rex_md_lag1 = all_RD['BTC'].shift(1)
rex_md_lag5 = all_RD['BTC'].rolling(window=5).mean().shift(1)
rex_md_lag22 = all_RD['BTC'].rolling(window=22).mean().shift(1)


# 数据准备 (从您新提供的数据结构)
model1 = pd.DataFrame({
    'RV': RV['RV'],
    'REX_m_lag1': rex_m_lag1,
    'REX_m_lag5': rex_m_lag5,
    'REX_m_lag22': rex_m_lag22,
    'REX_p_lag1': rex_p_lag1,
    'REX_p_lag5': rex_p_lag5,
    'REX_p_lag22': rex_p_lag22,
    'REX_md_lag1': rex_md_lag1,
    'REX_md_lag5': rex_md_lag5,
    'REX_md_lag22': rex_md_lag22,
    'BTC_lag1': final_output_rm['Dynamic_RM_lag1']
}).dropna()

# --- 2. 数据划分 (已修正和简化) ---
window_size = 1800
test_size = 300

# 这是一个更稳健的数据划分方式
train_data_full = model1.iloc[:-test_size]
initial_train_data = train_data_full.iloc[-window_size:]
test_data = model1.iloc[-test_size:].reset_index(drop=True)

# --- 2. 模型核心函数 (保持您原有的逻辑) ---
def tvtp_ms_har_log_likelihood(params, y, X, z, n_states=2, return_filtered_probs=False):
    n, k = len(y), X.shape[1]
    beta = params[:k * n_states].reshape(n_states, k)
    sigma = np.exp(params[k * n_states: k * n_states + n_states])
    a = params[k * n_states + n_states: k * n_states + 2 * n_states]
    b = params[k * n_states + 2 * n_states:]
    log_filtered_prob = np.zeros((n, n_states))
    pred_prob = np.ones(n_states) / n_states
    log_lik = 0.0
    mu_cache = np.dot(X, beta.T)
    for t in range(n):
        if t > 0:
            logit_11 = np.clip(a[0] + b[0] * z[t - 1], -30, 30);
            p11 = 1.0 / (1.0 + np.exp(-logit_11))
            logit_22 = np.clip(a[1] + b[1] * z[t - 1], -30, 30);
            p22 = 1.0 / (1.0 + np.exp(-logit_22))
            P = np.array([[np.clip(p11, 1e-6, 1 - 1e-6), 1 - np.clip(p11, 1e-6, 1 - 1e-6)],
                          [1 - np.clip(p22, 1e-6, 1 - 1e-6), np.clip(p22, 1e-6, 1 - 1e-6)]])
            filtered_prob_prev = np.exp(log_filtered_prob[t - 1] - np.max(log_filtered_prob[t - 1]))
            pred_prob = (filtered_prob_prev / filtered_prob_prev.sum()) @ P
        log_cond_dens = norm.logpdf(y[t], mu_cache[t], np.maximum(sigma, 1e-8))
        log_joint_prob = np.log(np.maximum(pred_prob, 1e-12)) + log_cond_dens
        max_log_prob = np.max(log_joint_prob)
        log_marginal_prob = max_log_prob + np.log(np.sum(np.exp(log_joint_prob - max_log_prob)))
        log_filtered_prob[t] = log_joint_prob - log_marginal_prob
        log_lik += log_marginal_prob
    if return_filtered_probs: return -log_lik, log_filtered_prob
    return -log_lik if np.isfinite(log_lik) else 1e10


### <<< 核心升级：更换优化器为 L-BFGS-B 并使用边界 >>> ###
def fit_tvtp_model(y, X, z, initial_params, bounds, n_states=2):
    # 使用 L-BFGS-B，它支持边界约束，是解决这类问题的最佳选择
    result = minimize(tvtp_ms_har_log_likelihood,
                      initial_params,
                      args=(y, X, z, n_states, False),
                      method='L-BFGS-B',  # 换成 L-BFGS-B
                      bounds=bounds,     # 传入边界！这是防止失败的关键！
                      options={'maxiter': 500, 'disp': False, 'ftol': 1e-7})

    return result


# 预测函数保持不变
def predict_tvtp_1_step(X_pred_features, z_for_P_matrix, last_filt_probs_norm, params, n_states=2):
    X_pred_with_const = np.insert(X_pred_features, 0, 1.0)
    k = len(X_pred_with_const)
    beta = params[:k * n_states].reshape(n_states, k)
    a = params[k * n_states + n_states: k * n_states + 2 * n_states]
    b = params[k * n_states + 2 * n_states:]
    mu = X_pred_with_const @ beta.T
    logit_11 = np.clip(a[0] + b[0] * z_for_P_matrix, -30, 30);
    p11 = np.clip(1.0 / (1.0 + np.exp(-logit_11)), 1e-6, 1 - 1e-6)
    logit_22 = np.clip(a[1] + b[1] * z_for_P_matrix, -30, 30);
    p22 = np.clip(1.0 / (1.0 + np.exp(-logit_22)), 1e-6, 1 - 1e-6)
    P_matrix = np.array([[p11, 1 - p11], [1 - p22, p22]])
    pred_state_probs = last_filt_probs_norm @ P_matrix
    prediction = np.sum(pred_state_probs * mu)
    return prediction


# --- 3. 滚动窗口预测核心逻辑 (已更新以使用L-BFGS-B) ---
def rolling_window_forecast(initial_data, test_data, initial_params, bounds):
    predictions, actuals = [], []
    rolling_data = initial_data.copy()

    # 修正：根据model1的定义，特征列为REX相关变量
    feature_cols = ['REX_m_lag1', 'REX_m_lag5', 'REX_m_lag22',
                    'REX_p_lag1', 'REX_p_lag5', 'REX_p_lag22',
                    'REX_md_lag1', 'REX_md_lag5', 'REX_md_lag22']

    current_params = initial_params.copy()
    failure_count = 0

    for i in tqdm(range(len(test_data)), desc="Rolling Forecast (L-BFGS-B Version)"):
        try:
            y_win = rolling_data['RV'].values
            X_win = sm.add_constant(rolling_data[feature_cols].values)
            z_win = rolling_data['BTC_lag1'].values

            # <<< 调用我们修改后的 fit 函数，并传递 bounds >>>
            fit_result = fit_tvtp_model(y_win, X_win, z_win,
                                        initial_params=current_params,
                                        bounds=bounds)

            if fit_result.success:
                current_params = fit_result.x
            else:
                failure_count += 1
                if i % 10 == 0:
                    print(f"\n警告: 第 {i} 次迭代优化失败。消息: {fit_result.message}. 将沿用旧参数。")

            # --- (函数的其余部分保持不变) ---
            _, final_log_probs = tvtp_ms_har_log_likelihood(current_params, y_win, X_win, z_win, 2, True)
            exp_last_log_probs = np.exp(final_log_probs[-1] - np.max(final_log_probs[-1]))
            last_filtered_prob_norm = exp_last_log_probs / np.sum(exp_last_log_probs)

            X_pred = test_data[feature_cols].iloc[i].values
            z_for_P = rolling_data['BTC_lag1'].iloc[-1]
            prediction = predict_tvtp_1_step(X_pred, z_for_P, last_filtered_prob_norm, current_params)

            predictions.append(prediction)
            actuals.append(test_data['RV'].iloc[i])

        except Exception as e:
            # 增加对错误的更详细打印
            import traceback
            print(f"\n严重错误在第 {i} 次迭代: {e}")
            traceback.print_exc()  # 打印详细的错误追踪信息
            predictions.append(np.nan)
            actuals.append(test_data['RV'].iloc[i])

        new_observation = test_data.iloc[i:i + 1]
        rolling_data = pd.concat([rolling_data.iloc[1:], new_observation], ignore_index=True)

    print(f"\n--- 滚动预测完成 ---")
    print(f"总计优化失败次数: {failure_count} / {len(test_data)} ({failure_count / len(test_data):.2%})")
    return predictions, actuals


# --- 4. 主程序入口 (已完全修正，并集成了L-BFGS-B的准备工作) ---
if __name__ == "__main__":
    print("--- 步骤 1: 生成智能初始参数和边界 ---")

    # 修正：根据您model1的定义，特征列为REX相关变量
    feature_cols = ['REX_m_lag1', 'REX_m_lag5', 'REX_m_lag22',
                    'REX_p_lag1', 'REX_p_lag5', 'REX_p_lag22',
                    'REX_md_lag1', 'REX_md_lag5', 'REX_md_lag22']
    n_states = 2
    k = len(feature_cols) + 1  # 特征数 + 1个常数项

    # --- 生成智能初始参数 (您的代码保持不变) ---
    y_init = initial_train_data['RV'].values
    X_init = sm.add_constant(initial_train_data[feature_cols].values)
    ols_model = sm.OLS(y_init, X_init).fit()
    n_params = k * n_states + n_states + n_states + n_states
    initial_params = np.zeros(n_params)
    initial_params[0:k] = ols_model.params * 0.8
    initial_params[k:2 * k] = ols_model.params * 1.2
    initial_params[2 * k:2 * k + n_states] = [np.log(np.std(y_init) * 0.8), np.log(np.std(y_init) * 1.2)]
    start_a = 2 * k + n_states
    initial_params[start_a:start_a + n_states] = [1.5, 1.5]
    start_b = start_a + n_states
    initial_params[start_b:start_b + n_states] = [0.0, 0.0]
    print("智能初始参数已生成。")

    # <<< 关键新增：为L-BFGS-B定义参数的合理边界！ >>>
    bounds = (
        # 状态0的beta系数 (k个)，无特定边界
            [(None, None)] * k +
            # 状态1的beta系数 (k个)，无特定边界
            [(None, None)] * k +
            # log(sigma) for each state: 限制波动率的对数在一个合理范围
            [(-10, 5)] * n_states +
            # a for each state (转移概率的常数项): 限制logit值防止溢出
            [(-20, 20)] * n_states +
            # b for each state (转移概率的斜率): 限制驱动变量的影响
            [(-10, 10)] * n_states
    )
    print("参数优化边界已定义。")

    print("\n--- 步骤 2: 开始执行一次完整的滚动窗口预测 ---")

    # <<< 关键修正：只调用一次滚动预测函数，并将bounds传递进去 >>>
    predictions, actuals = rolling_window_forecast(initial_train_data, test_data, initial_params, bounds)

    # --- 步骤 3: 计算并打印损失函数 ---

    # 将结果保存到CSV文件，这是一个好的实践
    output_csv_file = 'tvtp_har_re_rm.csv'
    pd.DataFrame({'Predicted_RV': predictions, 'Actual_RV': actuals}).to_csv(output_csv_file, index=False)
    print(f"预测结果已保存至 {output_csv_file}")


    # 您的损失函数计算代码 (稍作修改以提高稳健性)
    def compute_losses(pred, true):
        pred_arr = np.array(pred)
        true_arr = np.array(true)

        # 清除nan值
        valid_indices = ~np.isnan(pred_arr) & ~np.isnan(true_arr)
        pred_clean = pred_arr[valid_indices]
        true_clean = true_arr[valid_indices]

        if len(pred_clean) == 0:
            print("警告: 没有有效的预测值用于计算损失。")
            return {"MSE": np.nan, "QLIKE": np.nan, "MAE": np.nan, "RMSE": np.nan}

        mse = np.mean((pred_clean - true_clean) ** 2)
        mae = np.mean(np.abs(pred_clean - true_clean))
        rmse = np.sqrt(mse)

        # QLIKE需要正数
        qlike_indices = (pred_clean > 1e-9) & (true_clean > 1e-9)
        if not np.any(qlike_indices):
            qlike = np.nan
        else:
            qlike = np.mean(np.log(pred_clean[qlike_indices]) + true_clean[qlike_indices] / pred_clean[qlike_indices])

        return {"MSE": mse, "QLIKE": qlike, "MAE": mae, "RMSE": rmse}


    losses = compute_losses(predictions, actuals)
    print("\n--- 损失函数计算结果 ---")
    for loss_name, loss_value in losses.items():
        print(f"{loss_name}: {loss_value:.6f}")