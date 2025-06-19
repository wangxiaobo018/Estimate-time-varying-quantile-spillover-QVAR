# --- 步骤 1: 计算日收益率数据 ---
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from numba import jit
from concurrent.futures import ThreadPoolExecutor
import warnings


from scipy.stats import norm
from scipy.special import logsumexp
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

from statsmodels.tsa.vector_ar.var_model import VAR
import os
from scipy.stats import norm
from scipy.special import logsumexp
import json
from numpy.linalg import inv
import os

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


import os
from scipy.stats import norm
from scipy.special import logsumexp
import json
from numpy.linalg import inv

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # 用于显示进度条，建议安装: pip install tqdm



# --- 步骤 1: 数据准备 (保持不变) ---
def calculate_daily_returns(df, coin_name):
    """计算日收益率"""
    data = df[['time', 'close']].copy()
    data.columns = ['DT', 'PRICE']
    data = data.dropna()
    data['DT'] = pd.to_datetime(data['DT'])
    data['Date'] = data['DT'].dt.date
    daily_data = data.groupby('Date')['PRICE'].last().reset_index()
    daily_data.columns = ['DT', f'PRICE_{coin_name}']
    daily_data[f'RET_{coin_name}'] = (daily_data[f'PRICE_{coin_name}'] /
                                      daily_data[f'PRICE_{coin_name}'].shift(1) - 1) * 100
    daily_data = daily_data.dropna()[['DT', f'RET_{coin_name}']]
    return daily_data


# 修改为您的文件路径
data_files = {
    'BTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/BTCUSDT_5m.csv",
    'DASH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/DASHUSDT_5m.csv",
    'ETH': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/ETHUSDT_5m.csv",
    'LTC': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/LTCUSDT_5m.csv",
    'XLM': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/XLMUSDT_5m.csv",
    'XRP': "c:/Users/lenovo/Desktop/spillover/crypto_5min_data/XRPUSDT_5m.csv"
}

returns_dfs = []
for coin, file_path in data_files.items():
    try:
        df = pd.read_csv(file_path)
        returns_df = calculate_daily_returns(df, coin)
        if returns_df is not None:
            returns_dfs.append(returns_df)
    except Exception as e:
        print(f"处理 {coin} 时发生错误: {e}")

if returns_dfs:
    all_returns = returns_dfs[0]
    for returns_df_to_merge in returns_dfs[1:]:
        all_returns = all_returns.merge(returns_df_to_merge, on='DT', how='inner')

    all_returns = all_returns.set_index('DT').sort_index().dropna()
    column_mapping = {col: col.replace('RET_', '') for col in all_returns.columns if col.startswith('RET_')}
    all_returns = all_returns.rename(columns=column_mapping)
    print("--- 收益率数据准备完毕 ---")
    print(all_returns.head())
else:
    print("数据准备失败。")
    exit()

warnings.filterwarnings('ignore')



class TVP_QVAR_DY:
    """
    时变参数分位数向量自回归动态溢出模型
    Time-Varying Parameter Quantile VAR Dynamic Spillover Model
    """

    def __init__(self, data, var_names=None):
        """
        初始化模型
        """
        if isinstance(data, pd.DataFrame):
            self.data = data.values
            self.var_names = data.columns.tolist()
            self.dates = data.index
        else:
            self.data = data
            self.var_names = var_names if var_names else [f'Var{i + 1}' for i in range(data.shape[1])]
            self.dates = None

        self.n_vars = self.data.shape[1]
        self.n_obs = self.data.shape[0]

    def QVAR(self, p=1, tau=0.5):
        """
        分位数向量自回归模型估计
        """
        y = self.data
        k = self.n_vars
        coef_matrix = []
        residuals = []

        for i in range(k):
            # 构建滞后矩阵
            yx = self._embed(y, p + 1)
            y_dep = y[p:, i]
            x_indep = yx[:, k:]

            # 分位数回归
            try:
                qr_model = QuantReg(y_dep, x_indep)
                qr_result = qr_model.fit(q=tau, max_iter=10000, p_tol=1e-5)

                # 提取系数
                coef = qr_result.params
                coef_matrix.append(coef)

                # 计算残差
                res = qr_result.resid
                residuals.append(res)
            except Exception as e:
                print(f"Warning: 变量 {i} 的分位数回归失败: {e}")
                # 使用OLS作为备选
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression(fit_intercept=False)
                lr.fit(x_indep, y_dep)
                coef = lr.coef_
                coef_matrix.append(coef)
                res = y_dep - lr.predict(x_indep)
                residuals.append(res)

        # 计算残差协方差矩阵
        residuals = np.column_stack(residuals)
        Q = np.dot(residuals.T, residuals) / len(residuals)
        B = np.array(coef_matrix)

        return {'B': B, 'Q': Q}

    def _embed(self, y, dimension):
        """
        创建嵌入矩阵（类似R的embed函数）
        """
        n = len(y)
        k = y.shape[1]
        m = n - dimension + 1

        result = np.zeros((m, dimension * k))
        for i in range(m):
            for j in range(dimension):
                result[i, j * k:(j + 1) * k] = y[i + dimension - j - 1]

        return result

    def GFEVD(self, Phi, Sigma, n_ahead=10, normalize=True, standardize=True):
        """
        广义预测误差方差分解（修正版）
        """
        # 从伴随矩阵中提取VAR系数
        k = Sigma.shape[0]  # 变量个数

        # 如果Phi是伴随矩阵形式，提取前k行
        if Phi.shape[0] > k:
            Phi_reduced = Phi[:k, :]
        else:
            Phi_reduced = Phi

        # 计算脉冲响应
        A = self._tvp_Phi(Phi_reduced, n_ahead - 1)
        gi = np.zeros_like(A)
        sigmas = np.sqrt(np.diag(Sigma))
        sigmas[sigmas == 0] = 1e-10  # 避免除零

        for j in range(A.shape[2]):
            gi[:, :, j] = np.dot(np.dot(A[:, :, j], Sigma),
                                 np.linalg.inv(np.diag(sigmas))).T

        # 标准化
        if standardize:
            girf = np.zeros_like(gi)
            diag_gi = np.diag(gi[:, :, 0]).copy()
            diag_gi[diag_gi == 0] = 1
            for i in range(gi.shape[2]):
                girf[:, :, i] = gi[:, :, i] / diag_gi[:, np.newaxis]
            gi = girf

        # 计算FEVD
        num = np.sum(gi ** 2, axis=2)
        den = np.sum(num, axis=1)
        den[den == 0] = 1
        fevd = (num.T / den).T

        if normalize:
            row_sums = np.sum(fevd, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            fevd = fevd / row_sums

        return {'GFEVD': fevd, 'GIRF': gi}

    def _tvp_Phi(self, x, nstep=30):
        """
        计算时变VAR的脉冲响应函数（修正版）
        """
        K = x.shape[0]
        if x.shape[1] % K != 0:
            raise ValueError(f"系数矩阵维度不匹配: {x.shape}")

        p = x.shape[1] // K

        # 提取VAR系数矩阵
        A = np.zeros((K, K, p))
        for i in range(p):
            A[:, :, i] = x[:, i * K:(i + 1) * K]

        # 计算脉冲响应函数
        Phi = np.zeros((K, K, nstep + 1))
        Phi[:, :, 0] = np.eye(K)

        for i in range(1, nstep + 1):
            Phi[:, :, i] = np.zeros((K, K))
            for j in range(min(i, p)):
                Phi[:, :, i] += np.dot(Phi[:, :, i - j - 1], A[:, :, j])

        return Phi

    def DCA(self, CV, digit=2):
        """
        动态连通性分析（修正版）
        """
        k = CV.shape[0]
        CT = CV * 100  # 转换为百分比

        # 计算各项指标
        OWN = np.diag(np.diag(CT))
        TO = np.sum(CT - OWN, axis=0)
        FROM = np.sum(CT - OWN, axis=1)
        NET = TO - FROM
        TCI = np.mean(TO)
        NPSO = CT - CT.T
        NPDC = np.sum(NPSO > 0, axis=1)

        # 构建完整的结果表格（类似R代码的格式）
        # 首先是CT矩阵和FROM列
        table = np.column_stack([CT, FROM])

        # 添加TO行
        to_row = np.append(TO, np.sum(TO))

        # 添加Inc.Own行（包含自身影响的总影响）
        inc_own = np.append(np.sum(CT, axis=0), TCI)

        # 添加NET行
        net_row = np.append(NET, TCI)

        # 添加NPDC行
        npdc_row = np.append(NPDC, 0)

        # 组合所有行
        full_table = np.vstack([table, to_row, inc_own, net_row, npdc_row])

        return {
            'CT': CT,
            'TCI': TCI,
            'TCI_corrected': TCI * k / (k - 1),
            'TO': TO,
            'FROM': FROM,
            'NET': NET,
            'NPSO': NPSO,
            'NPDC': NPDC,
            'TABLE': full_table
        }

    def _create_RHS_NI(self, templag, r, nlag, t):
        """创建右侧变量矩阵（修正版）"""
        K = nlag * (r ** 2)
        x_t = np.zeros(((t - nlag) * r, K))

        for i in range(t - nlag):
            for eq in range(r):
                row_idx = i * r + eq
                col_start = 0

                for j in range(nlag):
                    xtemp = templag[i, j * r:(j + 1) * r]

                    for var in range(r):
                        col_idx = col_start + eq * r + var
                        x_t[row_idx, col_idx] = xtemp[var]

                    col_start += r * r

        Flag = np.vstack([np.zeros((nlag * r, K)), x_t])
        return Flag

    def KFS_parameters(self, Y, l, nlag, beta_0_mean, beta_0_var, Q_0):
        """
        卡尔曼滤波和平滑参数估计（修正版）
        """
        n = p = Y.shape[1]
        r = p
        m = nlag * (r ** 2)
        k = nlag * r
        t = Y.shape[0]

        # 初始化矩阵
        beta_pred = np.zeros((m, t))
        beta_update = np.zeros((m, t))
        Rb_t = np.zeros((m, m, t))
        Sb_t = np.zeros((m, m, t))
        beta_t = np.zeros((k, k, t))
        Q_t = np.zeros((r, r, t))

        # 衰减因子
        l_2, l_4 = l[1], l[3]

        # 构建滞后矩阵
        yy = Y[nlag:]
        templag = self._embed(Y, nlag + 1)[:, Y.shape[1]:]

        # 构建状态矩阵
        Flag = self._create_RHS_NI(templag, r, nlag, t)

        # 卡尔曼滤波
        for irep in range(t):
            if irep % 100 == 0:
                print(f"卡尔曼滤波进度: {irep}/{t}")

            # 更新Q矩阵
            if irep == 0:
                Q_t[:, :, irep] = Q_0
            elif irep > 0:
                if irep <= nlag:
                    Gf_t = 0.1 * np.outer(Y[irep], Y[irep])
                else:
                    idx = irep - nlag - 1
                    if idx < len(yy) and irep > 0:
                        B_prev = self._construct_B_matrix(beta_update[:, irep - 1], r, nlag)
                        y_pred = np.dot(templag[idx], B_prev[:r, :].T)
                        resid = yy[idx] - y_pred
                        Gf_t = np.outer(resid, resid)
                    else:
                        Gf_t = Q_t[:, :, irep - 1]

                Q_t[:, :, irep] = l_2 * Q_t[:, :, irep - 1] + (1 - l_2) * Gf_t[:r, :r]

            # 更新beta
            if irep <= nlag:
                beta_pred[:, irep] = beta_0_mean
                beta_update[:, irep] = beta_pred[:, irep]
                Rb_t[:, :, irep] = beta_0_var
            else:
                beta_pred[:, irep] = beta_update[:, irep - 1]
                Rb_t[:, :, irep] = (1 / l_4) * Sb_t[:, :, irep - 1]

            # 卡尔曼更新
            if irep >= nlag and (irep - 1) * r < Flag.shape[0]:
                try:
                    flag_slice = Flag[(irep - 1) * r:irep * r, :]
                    Rx = np.dot(Rb_t[:, :, irep], flag_slice.T)
                    KV_b = Q_t[:, :, irep] + np.dot(flag_slice, Rx)
                    KG = np.dot(Rx, np.linalg.pinv(KV_b))

                    if irep < t:
                        innovation = Y[irep] - np.dot(flag_slice, beta_pred[:, irep])
                        beta_update[:, irep] = beta_pred[:, irep] + np.dot(KG, innovation)
                        Sb_t[:, :, irep] = Rb_t[:, :, irep] - np.dot(KG, np.dot(flag_slice, Rb_t[:, :, irep]))
                except Exception as e:
                    print(f"Warning at time {irep}: {e}")
                    beta_update[:, irep] = beta_pred[:, irep]
                    Sb_t[:, :, irep] = Rb_t[:, :, irep]

            # 构建B矩阵
            B = self._construct_B_matrix(beta_update[:, irep], r, nlag)

            # 检查稳定性
            eigenvalues = np.linalg.eigvals(B)
            if np.max(np.abs(eigenvalues)) <= 1.1 or irep == 0:
                beta_t[:, :, irep] = B
            else:
                beta_t[:, :, irep] = beta_t[:, :, irep - 1] if irep > 0 else B
                beta_update[:, irep] = 0.99 * beta_update[:, irep - 1] if irep > 0 else beta_update[:, irep]

        return {'beta_t': beta_t, 'Q_t': Q_t}

    def _construct_B_matrix(self, beta_vec, r, nlag):
        """构建VAR的伴随矩阵"""
        k = nlag * r
        B = np.zeros((k, k))

        # 重塑beta向量
        beta_mat = beta_vec.reshape(r, -1)

        # 填充第一行块
        B[:r, :] = beta_mat

        # 添加单位矩阵部分（如果nlag > 1）
        if nlag > 1:
            B[r:, :r * (nlag - 1)] = np.eye(r * (nlag - 1))

        return B

    def run_analysis(self, nlag=1, nfore=10, tau=0.5,
                     l=[0.99, 0.99, 0.99, 0.96], window=None):
        """
        运行完整的TVP-QVAR-DY分析
        """
        results = {}

        # 1. 静态分析
        print("运行静态QVAR分析...")
        static_qvar = self.QVAR(p=nlag, tau=tau)
        static_gfevd = self.GFEVD(static_qvar['B'], static_qvar['Q'],
                                  n_ahead=nfore, normalize=True, standardize=True)
        static_dca = self.DCA(static_gfevd['GFEVD'])

        results['static'] = {
            'qvar': static_qvar,
            'gfevd': static_gfevd,
            'dca': static_dca
        }

        # 打印静态溢出矩阵
        print("\n静态溢出矩阵:")
        print(f"总连通性指数 (TCI): {static_dca['TCI']:.2f}%")

        # 2. 时变分析
        if window is None:
            print("\n运行时变参数估计...")
            # 初始化参数
            beta_0_mean = static_qvar['B'].flatten()
            beta_0_var = 0.05 * np.eye(len(beta_0_mean))
            Q_0 = static_qvar['Q']

            # 运行卡尔曼滤波
            kfs_results = self.KFS_parameters(self.data, l, nlag,
                                              beta_0_mean, beta_0_var, Q_0)

            # 计算动态溢出
            print("\n计算动态溢出指数...")
            t = self.n_obs
            total = np.zeros(t)
            gfevd = np.zeros((self.n_vars, self.n_vars, t))
            net = np.zeros((t, self.n_vars))
            to = np.zeros((t, self.n_vars))
            from_others = np.zeros((t, self.n_vars))
            npso = np.zeros((self.n_vars, self.n_vars, t))

            for i in range(t):
                if i % 100 == 0:
                    print(f"动态溢出计算进度: {100 * i / t:.2f}%")

                try:
                    # 计算GFEVD
                    gfevd_i = self.GFEVD(Phi=kfs_results['beta_t'][:, :, i],
                                         Sigma=kfs_results['Q_t'][:, :, i],
                                         n_ahead=nfore, standardize=True, normalize=True)
                    gfevd[:, :, i] = gfevd_i['GFEVD']

                    # 计算DCA
                    dca_i = self.DCA(gfevd[:, :, i])
                    to[i, :] = dca_i['TO']
                    from_others[i, :] = dca_i['FROM']
                    net[i, :] = dca_i['NET']
                    npso[:, :, i] = dca_i['NPSO']
                    total[i] = dca_i['TCI']
                except Exception as e:
                    if i % 100 == 0:  # 减少警告输出
                        print(f"Warning at time {i}: {e}")
                    # 使用前一期的值
                    if i > 0:
                        to[i, :] = to[i - 1, :]
                        from_others[i, :] = from_others[i - 1, :]
                        net[i, :] = net[i - 1, :]
                        npso[:, :, i] = npso[:, :, i - 1]
                        total[i] = total[i - 1]

            results['dynamic'] = {
                'total': total,
                'to': to,
                'from': from_others,
                'net': net,
                'npso': npso,
                'gfevd': gfevd
            }

        return results

    # 在你的 TVP_QVAR_DY 类中，用这个新版本替换原来的 plot_results 函数

    # 在你的 TVP_QVAR_DY 类中，用这个新版本替换原来的 plot_results 函数

    def plot_results(self, results, tau, dates=None):
        """
        绘制分析结果 (修改版 v2)
        - 图例统一固定在右上角
        """
        if dates is None:
            dates = self.dates if self.dates is not None else np.arange(self.n_obs)

        # --- 1. 绘制总溢出指数图 ---
        if 'dynamic' in results and 'total' in results['dynamic']:
            plt.figure(figsize=(12, 6))
            dynamic_results = results['dynamic']
            total_spillover = dynamic_results['total']

            if len(dates) > len(total_spillover):
                dates = dates[len(dates) - len(total_spillover):]

            # 给总溢出指数的填充区域也加上标签，以便图例显示

            plt.fill_between(dates, 0, total_spillover, color='red', alpha=0.4)

            plt.grid(True, linestyle='--', alpha=0.6)

            plt.ylabel('TCI (%)', fontsize=12)
            plt.xlabel('Date', fontsize=12)

            # --- 修改点 1 ---
            # 将图例位置固定在右上角
            plt.legend(loc='upper right')

            plt.tight_layout()

            pdf_filename_tci = f'total_spillover_tau_{str(tau).replace(".", "")}.pdf'
            plt.savefig(pdf_filename_tci, format='pdf', bbox_inches='tight')
            print(f"总溢出图已保存为: {pdf_filename_tci}")
            plt.show()

        # --- 2. 为每个变量绘制独立的净溢出图 ---
        if 'dynamic' in results and 'net' in results['dynamic']:
            net_spillover = results['dynamic']['net']

            if len(dates) > net_spillover.shape[0]:
                dates = dates[len(dates) - net_spillover.shape[0]:]

            for i in range(self.n_vars):
                var_name = self.var_names[i]
                plt.figure(figsize=(12, 6))
                net_series = net_spillover[:, i]



                plt.fill_between(dates, net_series, 0, where=(net_series >= 0),
                                 color='green', alpha=0.4, interpolate=True, label='NT (Net Transmitter)')
                plt.fill_between(dates, net_series, 0, where=(net_series < 0),
                                 color='red', alpha=0.4, interpolate=True, label='NR (Net Receiver)')

                plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
                plt.grid(True, linestyle='--', alpha=0.6)

                # --- 为了统一，将子图标题也加在这里 ---

                plt.ylabel('Net Spillover (%)', fontsize=12)  # Y轴标签也统一一下
                plt.xlabel('Date', fontsize=12)  # X轴标签也统一一下

                # --- 修改点 2 ---
                # 将图例位置固定在右上角
                plt.legend(loc='upper right')

                plt.tight_layout()

                pdf_filename_net = f'net_spillover_{var_name}_tau_{str(tau).replace(".", "")}.pdf'
                plt.savefig(pdf_filename_net, format='pdf', bbox_inches='tight')
                print(f"{var_name} 的净溢出图已保存为: {pdf_filename_net}")
                plt.show()

def main():

    model = TVP_QVAR_DY(all_returns)

    # --- 4. 循环遍历不同的分位数进行分析和绘图 ---
    quantiles_to_run = [0.05, 0.5, 0.95]

    for tau_val in quantiles_to_run:
        print(f"\n\n{'=' * 25}")
        print(f" 开始分析, 分位数 tau = {tau_val} ")
        print(f"{'=' * 25}\n")

        # 运行完整的时变分析
        results = model.run_analysis(nlag=4, nfore=10, tau=tau_val)

        # 绘制结果图表并保存为PDF
        print(f"\n--- 正在为 tau = {tau_val} 绘制并保存图表 ---")
        model.plot_results(results, tau=tau_val, dates=all_returns.index)

    print("\n\n所有分析和绘图已完成！")


if __name__ == "__main__":
    main()