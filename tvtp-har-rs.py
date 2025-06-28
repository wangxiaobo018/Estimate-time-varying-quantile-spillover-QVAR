import pandas as pd
import numpy as np
from pathlib import Path

from scipy.stats import norm

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


def calculate_returns(prices):
    """Calculate returns from price series."""
    returns = np.zeros(len(prices))
    for i in range(1, len(prices)):
        if prices[i - 1] == 0:
            returns[i] = np.nan
        else:
            returns[i] = ((prices[i] - prices[i - 1]) / prices[i - 1]) * 100
    returns[0] = 0
    return returns


def calculate_RS(data):
    """Calculate RS+ and RS- from returns."""
    positive_returns = np.where(data['Ret'] > 0, data['Ret'], 0)
    negative_returns = np.where(data['Ret'] < 0, data['Ret'], 0)

    RS_plus = np.sum(np.square(positive_returns))
    RS_minus = np.sum(np.square(negative_returns))

    return pd.Series({
        'RS_plus': RS_plus,
        'RS_minus': RS_minus
    })


def process_har_rs_model(data_idx_path, crypto_id):
    """Process data for HAR-RS model for a single cryptocurrency."""
    # Read the index data
    df_idx = pd.read_csv(data_idx_path)

    # Process index data
    data_ret_idx = (
        df_idx[['time', 'code', 'close']]
        .rename(columns={'time': 'DT', 'code': 'id', 'close': 'PRICE'})
        .dropna()
    )

    # Calculate returns for index data
    grouped_idx = data_ret_idx.groupby('id')
    returns_list_idx = []

    for name, group in grouped_idx:
        group_returns = pd.DataFrame({
            'DT': group['DT'],
            'id': group['id'],
            'Ret': calculate_returns(group['PRICE'].values)
        })
        returns_list_idx.append(group_returns)

    data_ret_idx = pd.concat(returns_list_idx, ignore_index=True)

    # Filter for specific index
    data_cj = data_ret_idx.query(f'id == "{crypto_id}"').copy()

    # Calculate RS statistics by date
    result = (
        data_cj.groupby(pd.to_datetime(data_cj['DT']).dt.date)
        .apply(calculate_RS)
        .reset_index()
    )

    # Rename columns for consistency
    result = result.rename(columns={'level_0': 'DT'})
    return result


def combine_rs_data(data_files):
    """Combine RS_plus and RS_minus for multiple cryptocurrencies."""
    rs_plus_list = []
    rs_minus_list = []

    for crypto_id, file_path in data_files.items():
        # Process each cryptocurrency
        result = process_har_rs_model(file_path, crypto_id)
        # Convert DT to datetime for consistency
        result['DT'] = pd.to_datetime(result['DT'])
        # Create DataFrames for RS_plus and RS_minus
        rs_plus = result[['DT', 'RS_plus']].rename(columns={'RS_plus': crypto_id})
        rs_minus = result[['DT', 'RS_minus']].rename(columns={'RS_minus': crypto_id})
        rs_plus_list.append(rs_plus)
        rs_minus_list.append(rs_minus)

    # Merge all RS_plus DataFrames on DT
    rs_plus_combined = rs_plus_list[0]
    for df in rs_plus_list[1:]:
        rs_plus_combined = rs_plus_combined.merge(df, on='DT', how='outer')

    # Merge all RS_minus DataFrames on DT
    rs_minus_combined = rs_minus_list[0]
    for df in rs_minus_list[1:]:
        rs_minus_combined = rs_minus_combined.merge(df, on='DT', how='outer')

    # Set DT as index
    rs_plus_combined.set_index('DT', inplace=True)
    rs_minus_combined.set_index('DT', inplace=True)

    return rs_plus_combined, rs_minus_combined


# Example usage
if __name__ == "__main__":
    data_files = {
        'BTC': "BTCUSDT_5m.csv",
        'DASH': "DASHUSDT_5m.csv",
        'ETH': "ETHUSDT_5m.csv",
        'LTC': "LTCUSDT_5m.csv",
        'XLM': "XLMUSDT_5m.csv",
        'XRP': "XRPUSDT_5m.csv"
    }

    # Process all cryptocurrencies and combine results
    rs_plus_df, rs_minus_df = combine_rs_data(data_files)

    print("\nRS_plus Combined Data Sample:")
    print(rs_plus_df.head())
    print("\nRS_minus Combined Data Sample:")
    print(rs_minus_df.head())

# ... (您上面所有的函数定义和 if __name__ == "__main__": 块)
# 在您的代码最后，已经生成了 rs_plus_df 和 rs_minus_df

# ----------------- 新的构建代码从这里开始 -----------------

# 确保DataFrame不为空且索引正确
if rs_plus_df.empty or rs_minus_df.empty:
    raise ValueError("生成的 rs_plus_df 或 rs_minus_df 为空，请检查数据处理流程。")

# --- 步骤 1: 为 RS_plus 构建动态驱动变量 ---
print("\n" + "=" * 50)
print("正在为 RS_plus 构建动态驱动变量...")

# 复制一份数据以避免修改原始DataFrame
data_plus = rs_plus_df.copy()

# 1a. 定义市场状态：基于 BTC 的上行波动率 (RS_plus)
# 我们使用 BTC 的 RS_plus 来定义市场的“上行情绪”状态
rv_quantiles_plus = data_plus['BTC'].quantile([0.05, 0.95])
print(f"基于 BTC RS_plus 的阈值: \n{rv_quantiles_plus}")

data_plus['market_state'] = pd.cut(data_plus['BTC'],
                                   bins=[-np.inf, rv_quantiles_plus[0.05], rv_quantiles_plus[0.95], np.inf],
                                   labels=['Bear_Plus', 'Normal_Plus', 'Bull_Plus'])

# 1b. 构建动态驱动变量 Dynamic_RS_plus_lag1
# 逻辑与之前完全相同：牛市用XRP，其他用XLM
data_plus['Dynamic_RS_plus_lag1'] = np.where(
    data_plus['market_state'] == 'Bull_Plus',
    data_plus['XRP'].shift(1),  # 牛市状态，使用滞后一期的 XRP 的 RS_plus
    data_plus['XLM'].shift(1)  # 熊市/正常状态，使用滞后一期的 XLM 的 RS_plus
)

# 1c. 整理并展示结果
final_output_plus = data_plus[['BTC', 'market_state', 'XLM', 'XRP', 'Dynamic_RS_plus_lag1']].dropna()

# 复制一份数据
data_minus = rs_minus_df.copy()
rv_quantiles_minus = data_minus['BTC'].quantile([0.05, 0.95])
print(f"基于 BTC RS_minus 的阈值: \n{rv_quantiles_minus}")

data_minus['market_state'] = pd.cut(data_minus['BTC'],
                                    bins=[-np.inf, rv_quantiles_minus[0.05], rv_quantiles_minus[0.95], np.inf],
                                    labels=['Calm_Minus', 'Normal_Minus', 'Panic_Minus'])
data_minus['Dynamic_RS_minus_lag1'] = np.where(
    data_minus['market_state'] == 'Panic_Minus',
    data_minus['XRP'].shift(1),  # 恐慌状态，使用滞后一期的 XRP 的 RS_minus
    data_minus['XLM'].shift(1)  # 平静/正常状态，使用滞后一期的 XLM 的 RS_minus
)
# 2c. 整理并展示结果
final_output_minus = data_minus[['BTC', 'market_state', 'XLM', 'XRP', 'Dynamic_RS_minus_lag1']].dropna()

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
    returns = (prices / prices.shift(1) - 1) * 100
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
      .apply(lambda x: np.sum(x ** 2))
      .reset_index())

# Ensure RV has the correct column names
RV.columns = ['DT', 'RV']
RV['DT'] = pd.to_datetime(RV['DT'])
RV = RV.set_index('DT')['RV']  # 将 'DT' 设为索引，提取 'RV' 列

rs_p_lag1 = rs_plus_df['BTC'].shift(1)
rs_p_lag5 = rs_plus_df['BTC'].rolling(window=5).mean().shift(1)
rs_p_lag22 = rs_plus_df['BTC'].rolling(window=22).mean().shift(1)
rs_m_lag1 = rs_minus_df['BTC'].shift(1)
rs_m_lag5 = rs_minus_df['BTC'].rolling(window=5).mean().shift(1)
rs_m_lag22 = rs_minus_df['BTC'].rolling(window=22).mean().shift(1)

model_data = pd.DataFrame({
    'RV': RV,
    'rs_p_lag1': rs_p_lag1,
    'rs_p_lag5': rs_p_lag5,
    'rs_p_lag22': rs_p_lag22,
    'rs_m_lag1': rs_m_lag1,
    'rs_m_lag5': rs_m_lag5,
    'rs_m_lag22': rs_m_lag22,
    'BTC_lag1': final_output_plus['Dynamic_RS_plus_lag1']
})
model1 = model_data.dropna()

# --- 2. 数据划分 ---
window_size = 1800
test_size = 300

train_data_full = model1.iloc[:-test_size]
initial_train_data = train_data_full.iloc[-window_size:]
test_data = model1.iloc[-test_size:].reset_index(drop=True)


# --- 3. 修改后的模型核心函数 (支持单个驱动变量) ---
def tvtp_ms_har_log_likelihood(params, y, X, z, n_states=2, return_filtered_probs=False):
    n, k = len(y), X.shape[1]
    beta = params[:k * n_states].reshape(n_states, k)
    sigma = np.exp(params[k * n_states: k * n_states + n_states])
    a = params[k * n_states + n_states: k * n_states + 2 * n_states]
    b = params[k * n_states + 2 * n_states:]  # 单个驱动变量系数

    log_filtered_prob = np.zeros((n, n_states))
    pred_prob = np.ones(n_states) / n_states
    log_lik = 0.0
    mu_cache = np.dot(X, beta.T)

    for t in range(n):
        if t > 0:
            # 转移概率现在只依赖于一个驱动变量
            logit_11 = np.clip(a[0] + b[0] * z[t - 1], -30, 30)
            p11 = 1.0 / (1.0 + np.exp(-logit_11))
            logit_22 = np.clip(a[1] + b[1] * z[t - 1], -30, 30)
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

    if return_filtered_probs:
        return -log_lik, log_filtered_prob
    return -log_lik if np.isfinite(log_lik) else 1e10


def fit_tvtp_model(y, X, z, initial_params, bounds, n_states=2):
    result = minimize(tvtp_ms_har_log_likelihood,
                      initial_params,
                      args=(y, X, z, n_states, False),
                      method='L-BFGS-B',
                      bounds=bounds,
                      options={'maxiter': 500, 'disp': False, 'ftol': 1e-7})
    return result


# 修改预测函数支持单个驱动变量
def predict_tvtp_1_step(X_pred_features, z_for_P, last_filt_probs_norm, params, n_states=2):
    X_pred_with_const = np.insert(X_pred_features, 0, 1.0)
    k = len(X_pred_with_const)
    beta = params[:k * n_states].reshape(n_states, k)
    a = params[k * n_states + n_states: k * n_states + 2 * n_states]
    b = params[k * n_states + 2 * n_states:]

    mu = X_pred_with_const @ beta.T

    # 转移概率计算只包含一个驱动变量
    logit_11 = np.clip(a[0] + b[0] * z_for_P, -30, 30)
    p11 = np.clip(1.0 / (1.0 + np.exp(-logit_11)), 1e-6, 1 - 1e-6)
    logit_22 = np.clip(a[1] + b[1] * z_for_P, -30, 30)
    p22 = np.clip(1.0 / (1.0 + np.exp(-logit_22)), 1e-6, 1 - 1e-6)

    P_matrix = np.array([[p11, 1 - p11], [1 - p22, p22]])
    pred_state_probs = last_filt_probs_norm @ P_matrix
    prediction = np.sum(pred_state_probs * mu)
    return prediction


# --- 4. 修改滚动窗口预测函数 ---
def rolling_window_forecast(initial_data, test_data, initial_params, bounds):
    predictions, actuals = [], []
    rolling_data = initial_data.copy()

    # 修改特征列名匹配
    feature_cols = [col for col in model1.columns if col.startswith('rs_')]

    current_params = initial_params.copy()
    failure_count = 0

    for i in tqdm(range(len(test_data)), desc="Rolling Forecast HAR-RS"):
        try:
            y_win = rolling_data['RV'].values
            X_win = sm.add_constant(rolling_data[feature_cols].values)
            z_win = rolling_data['BTC_lag1'].values  # 单个驱动变量

            fit_result = fit_tvtp_model(y_win, X_win, z_win,
                                        initial_params=current_params,
                                        bounds=bounds)

            if fit_result.success:
                current_params = fit_result.x
            else:
                failure_count += 1
                if i % 10 == 0:
                    print(f"\n警告: 第 {i} 次迭代优化失败。消息: {fit_result.message}. 将沿用旧参数。")

            _, final_log_probs = tvtp_ms_har_log_likelihood(current_params, y_win, X_win, z_win, 2, True)
            exp_last_log_probs = np.exp(final_log_probs[-1] - np.max(final_log_probs[-1]))
            last_filtered_prob_norm = exp_last_log_probs / np.sum(exp_last_log_probs)

            X_pred = test_data[feature_cols].iloc[i].values
            z_for_P = rolling_data['BTC_lag1'].iloc[-1]
            prediction = predict_tvtp_1_step(X_pred, z_for_P, last_filtered_prob_norm, current_params)

            predictions.append(prediction)
            actuals.append(test_data['RV'].iloc[i])

        except Exception as e:
            import traceback
            print(f"\n严重错误在第 {i} 次迭代: {e}")
            traceback.print_exc()
            predictions.append(np.nan)
            actuals.append(test_data['RV'].iloc[i])

        new_observation = test_data.iloc[i:i + 1]
        rolling_data = pd.concat([rolling_data.iloc[1:], new_observation], ignore_index=True)

    print(f"\n--- 滚动预测完成 ---")
    print(f"总计优化失败次数: {failure_count} / {len(test_data)} ({failure_count / len(test_data):.2%})")
    return predictions, actuals


# --- 5. 主程序入口 ---
if __name__ == "__main__":
    print("--- 步骤 1: 生成智能初始参数和边界 ---")

    feature_cols = [col for col in model1.columns if col.startswith('rs_')]
    n_states = 2
    k = len(feature_cols) + 1  # 特征数 + 1个常数项

    # 生成智能初始参数
    y_init = initial_train_data['RV'].values
    X_init = sm.add_constant(initial_train_data[feature_cols].values)
    ols_model = sm.OLS(y_init, X_init).fit()

    # 参数数量：k*n_states(beta) + n_states(sigma) + n_states(a) + n_states(b)
    n_params = k * n_states + n_states + n_states + n_states
    initial_params = np.zeros(n_params)

    initial_params[0:k] = ols_model.params * 0.8
    initial_params[k:2 * k] = ols_model.params * 1.2
    initial_params[2 * k:2 * k + n_states] = [np.log(np.std(y_init) * 0.8), np.log(np.std(y_init) * 1.2)]
    start_a = 2 * k + n_states
    initial_params[start_a:start_a + n_states] = [1.5, 1.5]
    start_b = start_a + n_states
    initial_params[start_b:start_b + n_states] = [0.0, 0.0]  # 单个驱动变量系数

    print("智能初始参数已生成。")

    # 定义边界（只有一个驱动变量的边界）
    bounds = (
            [(None, None)] * k +  # 状态0的beta系数
            [(None, None)] * k +  # 状态1的beta系数
            [(-10, 5)] * n_states +  # log(sigma)
            [(-20, 20)] * n_states +  # a (转移概率常数项)
            [(-10, 10)] * n_states  # b (驱动变量系数)
    )
    print("参数优化边界已定义。")

    print("\n--- 步骤 2: 开始执行滚动窗口预测 ---")
    predictions, actuals = rolling_window_forecast(initial_train_data, test_data, initial_params, bounds)

    # 保存结果
    output_csv_file = 'tvtp_har_rs111_results.csv'
    pd.DataFrame({'Predicted_RV': predictions, 'Actual_RV': actuals}).to_csv(output_csv_file, index=False)
    print(f"预测结果已保存至 {output_csv_file}")


    # 计算损失函数
    def compute_losses(pred, true):
        pred_arr = np.array(pred)
        true_arr = np.array(true)

        valid_indices = ~np.isnan(pred_arr) & ~np.isnan(true_arr)
        pred_clean = pred_arr[valid_indices]
        true_clean = true_arr[valid_indices]

        if len(pred_clean) == 0:
            print("警告: 没有有效的预测值用于计算损失。")
            return {"MSE": np.nan, "QLIKE": np.nan, "MAE": np.nan, "RMSE": np.nan}

        mse = np.mean((pred_clean - true_clean) ** 2)
        mae = np.mean(np.abs(pred_clean - true_clean))
        rmse = np.sqrt(mse)

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