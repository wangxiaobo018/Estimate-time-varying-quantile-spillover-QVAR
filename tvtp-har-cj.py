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
data_files = {
    'BTC': "BTCUSDT_5m.csv",
    'DASH': "DASHUSDT_5m.csv",
    'ETH': "ETHUSDT_5m.csv",
    'LTC': "LTCUSDT_5m.csv",
    'XLM': "XLMUSDT_5m.csv",
    'XRP': "XRPUSDT_5m.csv"
}
# 定义 get_RV_BV 函数
def get_RV_BV(data, alpha=0.05, times=True):
    idx = 100 if times else 1
    df = data.copy()

    df['datetime'] = pd.to_datetime(df['time'])
    df['day'] = df['datetime'].dt.date

    results = []
    for day, group in df.groupby('day'):
        group = group.sort_values('datetime')

        group['Ret'] = (np.log(group['close']) - np.log(group['close'].shift(1))) * idx

        group = group.dropna(subset=['Ret'])
        n = len(group)

        if n < 5:
            continue

        # 计算 RV
        RV = np.sum(group['Ret'] ** 2)

        # 计算 BV
        abs_ret = np.abs(group['Ret'])
        BV = (np.pi / 2) * np.sum(abs_ret.shift(1) * abs_ret.shift(-1).dropna())

        # 计算 TQ
        TQ_coef = n * (2 ** (2 / 3) * gamma(7 / 6) / gamma(0.5)) ** (-3) * (n / (n - 4))

        term1 = abs_ret.iloc[4:].values
        term2 = abs_ret.iloc[2:-2].values
        term3 = abs_ret.iloc[:-4].values

        min_len = min(len(term1), len(term2), len(term3))
        if min_len > 0:
            TQ = TQ_coef * np.sum((term1[:min_len] ** (4 / 3)) *
                                  (term2[:min_len] ** (4 / 3)) *
                                  (term3[:min_len] ** (4 / 3)))
        else:
            continue

        # Z_test
        Z_test = ((RV - BV) / RV) / np.sqrt(((np.pi / 2) ** 2 + np.pi - 5) *
                                            (1 / n) * max(1, TQ / (BV ** 2)))

        # 计算 JV 和 C_t
        q_alpha = norm.ppf(1 - alpha)
        JV = (RV - BV) * (Z_test > q_alpha)
        C_t = (Z_test <= q_alpha) * RV + (Z_test > q_alpha) * BV

        results.append({
            'day': day,
            'C_t': C_t,
            'JV': JV
        })

    result_df = pd.DataFrame(results)
    return result_df

# 创建字典来存储 C_t 和 JV 数据
c_t_dict = {}
jv_dict = {}

# 遍历每个数据文件，计算 HAR-CJ
for code, file_path in data_files.items():
    # 读取数据
    df = pd.read_csv(file_path)

    # 筛选数据
    data_filtered = df[df['code'] == code].copy()

    # 计算 HAR-CJ
    har_cj = get_RV_BV(data_filtered, alpha=0.05, times=True)

    # 将 day 转换为字符串，以便合并时使用
    har_cj['day'] = har_cj['day'].astype(str)

    # 存储 C_t 和 JV 数据
    c_t_dict[code] = har_cj[['day', 'C_t']].set_index('day')['C_t']
    jv_dict[code] = har_cj[['day', 'JV']].set_index('day')['JV']

# 转换为 DataFrame
ct = pd.DataFrame(c_t_dict)
jv = pd.DataFrame(jv_dict)


all_CT = ct.reset_index()  # Resets the 'day' index to a column
all_CT = all_CT.rename(columns={'day': 'DT'})  # Renames the 'day' column to 'DT'
print(all_CT)
all_JV = jv.reset_index()  # Resets the 'day' index to a column
all_JV = all_JV.rename(columns={'day': 'DT'})  # Renames the 'day' column to 'DT'
print(all_JV)


features = all_CT.drop(columns=['DT'])
data = features.to_numpy()

columns = ["BTC", "DASH", "ETH", "LTC", "XLM", "XRP"]
data_df = pd.DataFrame(data, columns=columns)
##  选择2最优
# 步骤1: 计算 BTC 波动率的 0.05 和 0.95 分位数作为阈值
rv_quantiles = data_df['BTC'].quantile([0.05, 0.95])

# 步骤2: 使用 pd.cut 根据阈值创建 'market_state' 列
data_df['market_state'] = pd.cut(data_df['BTC'],
                                 bins=[-np.inf, rv_quantiles[0.05], rv_quantiles[0.95], np.inf],
                                 labels=['Bear', 'Normal', 'Bull'])


# 步骤3: 使用嵌套的 np.where 构建动态驱动变量
data_df['Dynamic_RV_lag1'] = np.where(
    data_df['market_state'] == 'Bear',
    data_df['XRP'].shift(1),  # 熊市状态，使用滞后一期的 XRP
    np.where(
        data_df['market_state'] == 'Bull',
        data_df['XLM'].shift(1),  # 牛市状态，使用滞后一期的 XLM
        data_df['LTC'].shift(1)   # 正常状态，使用滞后一期的 LTC
    )
)

final_output = data_df[['BTC', 'market_state', 'LTC', 'XLM', 'XRP', 'Dynamic_RV_lag1']].dropna()



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
RV = (data_filtered
      .groupby('DT')['Ret']
      .apply(lambda x: np.sum(x**2))
      .reset_index())

# Ensure RV has the correct column names
RV.columns = ['DT', 'RV']
print(RV)
# Convert DT to datetime for consistency with har_cj
RV['DT'] = pd.to_datetime(RV['DT'])


JV_lag1 = all_JV['BTC'].shift(1)
C_t_lag1 = all_CT['BTC'].shift(1)
JV_lag5 = all_JV['BTC'].rolling(window=5).mean().shift(1)
C_t_lag5 = all_CT['BTC'].rolling(window=5).mean().shift(1)
JV_lag22 = all_JV['BTC'].rolling(window=22).mean().shift(1)
C_t_lag22 = all_JV['BTC'].rolling(window=22).mean().shift(1)


model_data= pd.DataFrame({
    'RV':RV['RV'],
    'Jv_lag1': JV_lag1,
    'Jv_lag5': JV_lag5,
    'Jv_lag22': JV_lag22,
    'C_t_lag1': C_t_lag1,
    'C_t_lag5': C_t_lag5,
    'C_t_lag22': C_t_lag22,
    'BTC_lag1': final_output['Dynamic_RV_lag1']
})
model1 = model_data.dropna()

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
def rolling_window_forecast(initial_data, test_data, initial_params, bounds):  # <<< 增加 bounds 参数
    predictions, actuals = [], []
    rolling_data = initial_data.copy()

    # 修正：根据model1的定义，特征列以'rv_'开头
    feature_cols = [col for col in model1.columns if col.startswith('rv_')]

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

    # 修正：根据您model1的定义，特征列应该是 'rv_' 开头
    feature_cols = [col for col in model1.columns if col.startswith('rv_')]
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
    output_csv_file = 'tvtp_har_cj_results.csv'
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