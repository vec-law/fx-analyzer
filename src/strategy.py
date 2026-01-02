import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils import save_df

import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils import save_df

SUPPORTED_STRATEGIES = [1]

def plot_strategy(df_test, strategy):
    instrument = df_test['instrument'].iloc[0]
    interval = df_test['interval'].iloc[0]

    plt.figure(figsize=(12, 5), facecolor='white')

    plt.plot(
        df_test.index,
        df_test['target'],
        color='dimgray',
        linewidth=1.2,
        alpha=0.7, marker='o',
        markersize=3,
        markerfacecolor='dimgray',
        label='Cena zamknięcia'
    )

    plt.scatter(
        df_test.index[df_test['delta'] > 0],
        df_test.loc[df_test['delta'] > 0, 'predictions'],
        color='gold',
        s=3,
        alpha=0.7,
        zorder=2, label='Trend UP'
    )

    plt.scatter(
        df_test.index[df_test['delta'] < 0],
        df_test.loc[df_test['delta'] < 0, 'predictions'],
        color='indigo',
        s=3,
        alpha=0.7,
        zorder=2,
        label='Trend DOWN'
    )

    legend = "LEGENDA:"
    suffix = ''

    if strategy is not None and strategy in SUPPORTED_STRATEGIES:
        suffix = f"_{strategy}"

        m_style = dict(marker='o', markersize=6, markerfacecolor='white', markeredgecolor='dodgerblue', markeredgewidth=1.5, zorder=5)
        plt.plot([], [], color='green', linestyle=':', label='Transakcja zyskowna BUY', **m_style)
        plt.plot([], [], color='red', linestyle=':', label='Transakcja stratna BUY', **m_style)
        plt.plot([], [], color='green', linestyle='-.', label='Transakcja zyskowna SELL', **m_style)
        plt.plot([], [], color='red', linestyle='-.', label='Transakcja stratna SELL', **m_style)

        for i, r in df_test.dropna(subset=['buy_open', 'idx_close']).iterrows():
            plt.plot(
                [i, r['idx_close']],
                [r['buy_open'], r['buy_close']],
                color='green' if r['buy_diff'] >= 0 else 'red',
                linestyle=':',
                linewidth=2.0,
                **m_style
            )
            
        for i, r in df_test.dropna(subset=['sell_open', 'idx_close']).iterrows():
            plt.plot(
                [i, r['idx_close']],
                [r['sell_open'], r['sell_close']], 
                color='green' if r['sell_diff'] >= 0 else 'red',
                linestyle='-.',
                linewidth=2.5,
                **m_style
            )
        
        buy_result = df_test['buy_diff'].sum()
        sell_result = df_test['sell_diff'].sum()
        total_result = buy_result + sell_result

        legend = (
            f"LEGENDA:\nStrategia nr {strategy}\n"
            f"Wynik BUY: {buy_result:.2f}%\n"
            f"Wynik SELL: {sell_result:.2f}%\n"
            f"Wynik razem: {total_result:.2f}%"
        )

    plt.grid(True, alpha=0.15, color='gray')
    plt.ylabel(instrument + ' ' + interval)
    plt.xlabel(f"Numer próbki testowej (1 próbka = {interval})")
    
    plt.legend(
        loc='upper left', 
        fontsize='small', 
        title=legend,           
        title_fontsize='medium',   
        scatterpoints=5, 
        handlelength=4,
        frameon=True,              
        facecolor='white',         
        edgecolor='gray'           
    )

    plt.gca().get_legend()._legend_box.align = "left"
    plt.setp(plt.gca().get_legend().get_title(), fontsize='small', horizontalalignment='left')

    plt.tight_layout()

    plot_dir = os.path.join('data', 'img')
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    
    plot_filename = os.path.join(plot_dir, f"{instrument}_{interval}{suffix}.png")
    
    plt.savefig(plot_filename, dpi=300)
    print(f"   [plot_strategy] Wykres zapisano w pliku data/img/{instrument}_{interval}{suffix}.png")

    plt.show()

    return df_test

def save_transactions(df_test, strategy):
    instrument = df_test['instrument'].iloc[0]
    interval = df_test['interval'].iloc[0]

    mask = df_test['buy_open'].notna() | df_test['sell_open'].notna()
    cols = ['instrument', 'buy_open', 'buy_close', 'buy_diff', 'sell_open', 'sell_close', 'sell_diff']
    
    save_df(df_test[mask][cols], 'tran', f"{instrument}_{interval}_{strategy}.csv")
    print(f"   [save_trades] Zapisano transakcje w pliku /data/tran/{instrument}_{interval}_{strategy}.csv")

    return df_test

def calculate_results(df_test):
    if df_test is None or 'idx_close' not in df_test: return None
    if 'buy_open' not in df_test or 'buy_close' not in df_test: return None
    if 'sell_open' not in df_test or 'sell_close' not in df_test: return None

    df_test['buy_diff'] = np.nan
    df_test['sell_diff'] = np.nan

    mask_buy = df_test['buy_open'].notna() & df_test['idx_close'].notna()
    df_test.loc[mask_buy, 'buy_diff'] = ((df_test['buy_close'] - df_test['buy_open']) / df_test['buy_open']) * 100.0
    print(f"   [calculate_results] Wyliczenie df_test[buy_diff]")
    
    mask_sell = df_test['sell_open'].notna() & df_test['idx_close'].notna()
    df_test.loc[mask_sell, 'sell_diff'] = ((df_test['sell_open'] - df_test['sell_close']) / df_test['sell_open']) * 100.0
    print(f"   [calculate_results] Wyliczenie df_test[sell_diff]")

    print(f"   [calculate_results] Wynik buy [%]: {(df_test['buy_diff'].sum()):.2f}")
    print(f"   [calculate_results] Wynik sell [%]: {(df_test['sell_diff'].sum()):.2f}")
    print(f"   [calculate_results] Wynik całkowity [%]: {(df_test['buy_diff'].sum() + df_test['sell_diff'].sum()):.2f}")

    return df_test

def set_positions(df_test, strategy):
    df_test['buy_open'] = np.nan
    df_test['buy_close'] = np.nan
    df_test['sell_open'] = np.nan
    df_test['sell_close'] = np.nan
    df_test['idx_close'] = np.nan

    open_buys = []
    open_sells = []
    
    last_row_idx = df_test.index[-1]

    for i in range(len(df_test)):
        current_idx = df_test.index[i]

        if strategy == 1:
            if i < 3: continue
            delta = df_test.iloc[i]['delta']
            delta_minus_1 = df_test.iloc[i-1]['delta']
            delta_minus_2 = df_test.iloc[i-2]['delta']
            delta_minus_3 = df_test.iloc[i-3]['delta']
            delta_minus_4 = df_test.iloc[i-4]['delta']
            target = df_test.at[current_idx, 'target']
            diff_rel = df_test.iloc[i]['diff_rel']

            if  (delta < 0 and delta_minus_1 < 0  and delta_minus_2 < 0 and diff_rel > -0.015) or \
                (delta < 0 and delta_minus_1 < 0 and delta_minus_2 < 0 and delta_minus_3 < 0 and delta_minus_4 < 0) or \
                (diff_rel > 0.035) or \
                (current_idx == last_row_idx):
                for b_idx in open_buys:
                    df_test.at[b_idx, 'idx_close'] = current_idx
                    df_test.at[b_idx, 'buy_close'] = target
                open_buys = []

            if  (delta > 0 and delta_minus_1 > 0 and delta_minus_2 > 0 and diff_rel < 0.015) or \
                (delta > 0 and delta_minus_1 > 0 and delta_minus_2 > 0 and delta_minus_3 > 0 and delta_minus_4 > 0) or \
                (diff_rel < -0.035) or \
                (current_idx == last_row_idx):
                for s_idx in open_sells:
                    df_test.at[s_idx, 'idx_close'] = current_idx
                    df_test.at[s_idx, 'sell_close'] = target
                open_sells = []

            if (delta > 0 and diff_rel < 0.0005) or (delta > 0 and delta_minus_1 < 0):
                df_test.at[current_idx, 'buy_open'] = target
                open_buys.append(current_idx)

            if (delta < 0 and diff_rel > -0.0005) or (delta < 0 and delta_minus_1 > 0):
                df_test.at[current_idx, 'sell_open'] = target
                open_sells.append(current_idx)

        else: 
            print(f"Błąd: Nieprawidłowa strategia {strategy}")
            return None

    print(f"   [set_positions] Wyznaczono otwarcia/zamknięcia pozycji dla strategii {strategy}")
    return df_test

def add_indicators(df_test):
    near_zero = 1e-9

    df_test.loc[:, 'delta'] = df_test['predictions'].diff().fillna(0)
    print(f"   [add_indicators] Obliczono df_test[delta]")

    df_test.loc[:, 'diff_rel'] = (df_test['target'] - df_test['predictions']) / (df_test['predictions'] + near_zero)
    print(f"   [add_indicators] Obliczono df_test[diff_rel]")

    return df_test

def simulate_strategy(df_dict, strategy=None):

    df_test = df_dict['test'].copy()

    if(df_test := add_indicators(df_test)) is None: return None
    
    if strategy is not None:
        if strategy in SUPPORTED_STRATEGIES:
            if(df_test := set_positions(df_test, strategy)) is None: return None
            if(df_test := calculate_results(df_test)) is None: return None
            if(df_test := save_transactions(df_test, strategy)) is None: return None
        else:
            print(f"Nieprawidłowa strategia: {strategy}")
    
    if(df_test := plot_strategy(df_test, strategy)) is None: return None

    df_dict['test'] = df_test
    
    return df_dict
