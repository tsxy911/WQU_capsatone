import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import spearmanr,pearsonr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import copy
import matplotlib.pyplot as plt
from IPython.display import Markdown
from IPython.display import display, HTML
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
import seaborn as sns
import taosws
import handle_data_crypto
import warnings

# 设置警告级别为"ignore"，以忽略所有警告
warnings.filterwarnings("ignore")

upc = '#EDC251'
dnc = '#31598D'
fzc = '#D76046'
dfz = '#D9AE89'
sfz = '#728995'
adc = '#4C8CC0'

# 读取因子文件
def load_fac_data(start, end, file, factor_col):
    fac_data = pd.read_pickle(file)
    fac_data = fac_data.loc[:,['time','code',factor_col]]
    fac_data = fac_data[(fac_data['time'] >= start) & (fac_data['time'] <= end)]
    fac_data['factor'] = fac_data.groupby('code')[factor_col].shift(1)
    fac_data = fac_data.dropna()
    # fac_data['time'] = pd.to_datetime(fac_data['time'])
    fac_data = fac_data.pivot(index='time', columns='code', values='factor')
    return fac_data

def td(fac_data, asset, kline):
    start = min(fac_data.index)
    end = max(fac_data.index)
    codes = tuple(fac_data.columns)
    conn = taosws.connect("taosws://root:Wbp6Jfwx@localhost:6041/crpto_lf")
    sql = "select open_time, symbol, close, open FROM crpto_lf_{} WHERE intv = '{}' AND open_time >= '{}' AND open_time <= '{}' AND symbol IN {}"
    sql = sql.format(asset, kline, start, end, codes)
    result = conn.query(sql)
    columns = [x.name() for x in result.fields]
    hq = pd.DataFrame([d for d in result], columns=columns)
    hq.columns = ['time','code','close','open']
    hq = hq.sort_values(by = ['time','code'])
    hq = hq.reset_index(drop = True)
    hq['time'] = hq['time'].str[:19]
    return hq

def load_hq(asset, kline, skip_st, skip_bj):
    hqfile = asset+kline+'hq.pkl'
    hq = pd.read_pickle(hqfile)
    if skip_st == True:
        hq = hq[hq.is_st == False]
    if skip_bj == True:
        hq = hq[~hq['code'].str[:2].str.contains('68')]
    hq = hq.sort_values(by = ['time','code'])
    return hq

def get_hq(data, hq, price):
    start = str(min(data.index))
    end = str(max(data.index))
    try:
        hq = hq[(hq['time'] >= start) & (hq['time'] <= end)]
        hq = hq.sort_values(by = ['time','code'])
        hq['pre'] = hq.groupby('code')[price].shift(1)
        hq['ret'] = hq[price]/hq['pre'] - 1
        hq = hq.pivot(index='time', columns='code', values='ret')
        data.index = data.index.astype(str)
        hq = hq[hq.index.isin(list(data.index))]
        data = data.reindex(index=data.index, columns=hq.columns, fill_value=None)
        return hq, data
    except Exception as e:
        print('没有找到行情文件',str(e))
        
def pltline(ret, df, x, y, title):
    plt.figure(figsize=(16, 6))
    plt.title('Group Return Rate')
    for column in df.columns:
        if column == 'abs':
            df[column].plot(figsize=(16, 6),color = fzc, linestyle = '--',linewidth = 3,legend = True)
        else:
            df[column].plot(figsize=(16, 6),linewidth = 1, legend = True)
    plt.show()
    
    # 计算每个月每个股票的累计收益率
    ret.index = pd.to_datetime(ret.index)
    monthly_cumulative_ret = ret.resample('D').sum()
    monthly_cumulative_ret.index = pd.to_datetime(monthly_cumulative_ret.index).to_period('D')

    # 获取列名
    stock_codes = monthly_cumulative_ret.columns.tolist()

    base_color = adc
    custom_colors = [base_color + hex(int(255 * alpha))[2:].zfill(2) for alpha in np.linspace(0.2, 1, 5)]


    # 创建一个热力图
    plt.figure(figsize=(16, len(monthly_cumulative_ret)/2))
    sns.heatmap(monthly_cumulative_ret, annot=True, fmt=".4f", cmap='vlag', linewidths=0.5, cbar=False, annot_kws={'size': 10})

    # 设置x轴刻度位置和标签位置
    plt.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)

    # 显示图表
    plt.show()
    
def benchmark(data, hq):
    data[data.notna()] = 1
    benchmark = data*hq
    benchmark['ret'] = benchmark.mean(axis=1)
    benchmark['acc'] = (benchmark['ret']+1).cumprod()
    accdata = benchmark.loc[:,'acc']
    retdata = benchmark.loc[:,'ret']
    return accdata, retdata

def IC(data, period, hq):
    display(Markdown("---"))
    text = f"#### IC Analysis"
    display(Markdown(text))
    IC = pd.DataFrame(index = data.index, columns = [])
    IC_stata = pd.DataFrame(index = period, columns = ['ic_mean','ic_std'])
    for pn in range(len(period)):
        p = period[pn]
        for i in range(len(hq)-p):
            icc = data.iloc[i].corr(hq.iloc[i+p], method='spearman')
            IC.loc[IC.index[i],f'IC_{p}'] = icc
            
        # IC平均值、波动率和终值记录
        IC_stata.loc[p,:] = [IC[f'IC_{p}'].mean(),IC[f'IC_{p}'].std()]
        
        # 第一个、最后一个和中间一个lag画时序图
        if pn == 0 or pn == len(period) - 1 or pn == int((len(period) - 1)/2):
            column = f'IC_{p}'
        # for column in IC.columns:
            plt.figure(figsize=(16, 6))
            plt.title(column+'_lag')
            plt.axhline(y=0, color=fzc, linestyle='--')
            IC[column].plot(color = adc, linewidth = 0.3, legend = True, label = 'IC')
            IC[column].rolling(window=30).mean().plot(color = upc, legend = True, label = 'IC-30-mean')
            IC[column].cumsum().plot(secondary_y=True, color=fzc, legend = True, label = 'cum IC（right）')
            plt.show()
            
    # 绘制均值衰减和波动熵
    plt.figure(figsize=(16, 6)) 
    plt.title('IC mean decay and fluctuation entropy analysis')
    IC_stata['ic_mean'].plot(color = adc, linewidth = 0.3, legend = True, label = 'IC mean')
    IC_stata['ic_std'].plot(secondary_y=True, color=fzc, legend = True, label = 'IC volativity（right）')
    plt.show()
    
# 计算多空收益和费后多空收益
def lsf(labels, hq, fee_rate, abstype, num_groups):
    if abstype == 'bb':
        long_short = np.where(labels == 1, 1, np.where(labels == num_groups, -1, 0))
        long_short = pd.DataFrame(long_short, columns=labels.columns, index=labels.index)
        re = hq*long_short
        re['ret'] = re[re!=0].mean(axis=1)
        ret = re.drop(columns=['ret'])
        ret['fee'] = ret.diff().abs().sum(axis=1)
        ret['ret'] = re['ret']
        re = ret
        del ret
        re['feeed_ret'] = re['ret'] - re['fee']*fee_rate
        re['acc'] = (re['ret']+1).cumprod()
        re['feeed_acc'] = (re['feeed_ret']+1).cumprod()
    elif abstype == 'sb':
        long_short = np.where(labels == 1, -1, np.where(labels == num_groups, 1, 0))
        long_short = pd.DataFrame(long_short, columns=labels.columns, index=labels.index)
        re = hq*long_short
        re['ret'] = re[re!=0].mean(axis=1)
        ret = re.drop(columns=['ret'])
        ret['fee'] = ret.diff().abs().sum(axis=1)
        ret['ret'] = re['ret']
        re = ret
        del ret
        re['feeed_ret'] = re['ret'] - re['fee']*fee_rate
        re['acc'] = (re['ret']+1).cumprod()
        re['feeed_acc'] = (re['feeed_ret']+1).cumprod()
    plt.figure(figsize=(16, 6))
    plt.title('L-S ret before & after fee')
    re['acc'].plot(color = fzc, legend = True, label = 'L-S ret before fee (prod)')
    re['feeed_acc'].plot(color = dnc, legend = True, label = 'L-S ret after fee（prod）')
    plt.show()
    
# 默认当前时间点因子计算截面至上一个时间点，收益率是当前时间点close减去上一个时间点close
# 输入因子是矩阵形态
def factor_test(fac_data, factor_name, data_source, start, end, num_groups, fee_rate, rebalance_period, asset, kline, price, abstype, period, file):
    text = f"# **Factor Test Report**"
    context = f"Factor Name：{factor_name} | Type：{asset} |  Granularity：{kline} | Rebalance_Period：{rebalance_period} | Direct：{abstype}"
    timeline = f"Start Time：{start} -- End Time：{end}"
    display(Markdown(text))
    display(Markdown("---"))
    display(Markdown(context))
    display(Markdown(timeline))
    
    if data_source == 'local':
        full_hq = load_hq(asset, kline)
    elif data_source == 'td':
        full_hq = td(fac_data, asset, kline)
        
    if len(full_hq) > 0:
        data = fac_data
        hq = full_hq
        if rebalance_period != 1:
            data = data.iloc[::rebalance_period]
            
        display(Markdown("---"))
        text = f"### Group Test"
        display(Markdown(text))
        
        hq, data = get_hq(data,hq,price)
        accdata, retdata = benchmark(data.copy(), hq.copy())
        # 开始排名，值越大，分组号越小
        ranked_df = data.rank(axis=1, method='first', na_option='keep', ascending=False)
        labels = ranked_df.apply(lambda row: row.dropna().argsort().argsort() // (len(row.dropna()) / num_groups) +1 if not row.dropna().empty else np.nan, axis=1)
        del ranked_df
        for r in range(0,num_groups):
            g = r+1
            acc = f'group_{g}'+'_acc'
            ret = f'group_{g}'+'_ret'
            group = labels.where(labels == g, 0)
            group = group.where(group == 0, 1)
            group = group*hq
            group[ret] = group[group != 0].mean(axis=1)
            group[acc] = (group[ret]+1).cumprod()
            groupacc = group.loc[:,[acc]]
            accdata = pd.concat([accdata, groupacc],axis = 1)
            groupret = group.loc[:,[ret]]
            retdata = pd.concat([retdata, groupret],axis = 1)
            
        # 绘制多空收益曲线
        if abstype == 'bb':
            accdata['abs'] = accdata[f'group_{1}'+'_acc'] - accdata[f'group_{num_groups}'+'_acc'] + 1
            accdata['alpha'] = accdata[f'group_{1}'+'_acc'] - accdata['acc'] + 1
            retdata['abs'] = retdata[f'group_{1}'+'_ret'] - retdata[f'group_{num_groups}'+'_ret']
        else:
            accdata['abs'] = - accdata[f'group_{1}'+'_acc'] + accdata[f'group_{num_groups}'+'_acc'] + 1
            accdata['alpha'] = accdata[f'group_{num_groups}'+'_acc'] - accdata['acc'] + 1
            retdata['abs'] = - retdata[f'group_{1}'+'_ret'] + retdata[f'group_{num_groups}'+'_ret']
            
        plt.figure(figsize=(16, 6))
        plt.title('Long-Short Ret & Whole Pool Ret')
        accdata['abs'].plot(color = fzc, legend = True, label = 'L-S ret line (cumsum)')
        accdata['acc'].plot(color = dfz, legend = True, label = 'Whole Pool ret line')
        accdata['alpha'].plot(color = adc, legend = True, label = 'Best Group Alpha')
        plt.show()

        # 计算多空收益费后曲线
        lsf(labels, hq, fee_rate, abstype, num_groups)

        # 绘制分组收益率曲线
        del accdata['acc']
        pltline(retdata, accdata, 'time', 'cum ret', 'grouped cum ret')

        # 计算IC
        IC(data, period, hq)