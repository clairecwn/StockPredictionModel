"""Plain-English names for the inputs behind a forecast, so the page can show what pushed each call.

describe(feature, value, news_kind) returns a short phrase, or None for inputs that make no sense on their own.
Numbers are only quoted when the feature's units are certain; otherwise the phrase names the input.
"""
import numpy as np

KIND = {"earnings_guidance": "an earnings report", "analyst": "an analyst's change of view", "leadership": "a change in leadership",
        "legal_regulatory": "legal or regulatory news", "accounting_fraud": "an accounting issue", "activist_short": "an activist or short seller",
        "credit_distress": "credit news", "operational": "operational news", "politics_policy": "policy news", "ma_strategy": "deal news",
        "capital": "a capital raise", "other": "company news"}


def pct(x):
    return f"{abs(100 * x):.1f}%"


def times(x):
    return f"{x:.1f} times"


def daily(annual):
    return f"{100 * annual / np.sqrt(252):.1f}%"


def describe(f, x, kind="other", quiet=False):
    up = x >= 0
    if f.startswith("cls_"):
        return f"The kind of news: {KIND.get(kind, 'company news')}"
    if f.startswith("has_"):
        return "How often it has had this kind of news before"
    if quiet:
        Q = {
            "ar_1": f"{'Rose' if up else 'Fell'} {pct(x)} against the market today",
            "rev_5": f"{'Rose' if up else 'Fell'} {pct(x)} against the market this week",
            "ar_20": f"{'Up' if up else 'Down'} {pct(x)} against the market this month",
            "ar_60": f"{'Up' if up else 'Down'} {pct(x)} against the market over three months",
            "mom_250_20": f"{'Up' if up else 'Down'} {pct(np.expm1(x))} over the past year",
            "hi_52": f"{pct(x)} below its 52-week high",
            "lo_52": f"{pct(x)} above its 52-week low",
            "ma50_gap": f"{pct(x)} {'above' if up else 'below'} its 50-day average",
            "ma200_gap": f"{pct(x)} {'above' if up else 'below'} its 200-day average",
            "rsi_14": f"RSI of {x:.0f} ({'stretched up' if x > 70 else 'stretched down' if x < 30 else 'neither stretched up nor down'})",
            "lv_5_60": f"Swings this week are {times(np.exp(x))} its usual size",
            "lv_20_250": f"Swings this month are {times(np.exp(x))} the yearly norm",
            "vol_60": f"Usually moves about {daily(x)} a day",
            "abvol_5": f"Trading this week is {times(np.exp(x))} its usual volume",
            "vol_chg": f"Trading this month is {times(np.exp(x))} the yearly norm",
            "max_20": f"Biggest one-day rise this month: {pct(x)}",
            "min_20": f"Biggest one-day fall this month: {pct(x)}",
            "mkt_5": f"The market is {'up' if up else 'down'} {pct(x)} this week",
            "mkt_20": f"The market is {'up' if up else 'down'} {pct(x)} this month",
            "mkt_vol20": "How jumpy the whole market has been this month",
        }
        return Q.get(f)
    E = {
        "car_0_1": f"{'Rose' if up else 'Fell'} {pct(x)} more than the market on the news",
        "ar_day0_sd": f"The news-day move was {times(abs(x))} its usual daily swing",
        "abn_volume_0_1": f"Traded {times(x)} its usual volume on the news",
        "k": f"Day {int(x)} after the news",
        "cum_k": f"{'Up' if up else 'Down'} {pct(x)} against the market since the news",
        "cum_k_z": f"{'Up' if up else 'Down'} {times(abs(x))} its usual swing since the news",
        "car01_plus_cum": f"{'Up' if up else 'Down'} {pct(x)} against the market in total since the news",
        "above_low": f"{pct(x)} above its lowest point since the news",
        "above_low_z": "How far it has bounced off its lowest point since the news",
        "days_since_low": f"{int(x)} trading days since its lowest point after the news",
        "ar_k": f"{'Rose' if up else 'Fell'} {pct(x)} against the market today",
        "ar_k_z": "Today's move compared with its usual swing",
        "ar_last3_z": "How it moved over the last three days",
        "vol_rel_k": f"Trading today is {times(np.exp(x))} its usual volume",
        "vol_decay_k": f"Trading today is {times(np.exp(x))} the news-day level",
        "vol_rel3": "Trading over the last three days compared with normal",
        "mkt_k": f"The market {'rose' if up else 'fell'} {pct(x)} today",
        "mkt_cum_k": f"The market is {'up' if up else 'down'} {pct(x)} since the news",
        "mkt_day0": f"The market {'rose' if up else 'fell'} {pct(x)} on the news day",
        "vix": f"Market fear (VIX) at {x:.0f} on the news day",
        "vix_k": f"Market fear (VIX) at {x:.0f}",
        "vix_chg_k": f"Market fear (VIX) {'up' if up else 'down'} {abs(x):.1f} points since the news",
        "dist_52w_high": f"{pct(x)} below its 52-week high before the news",
        "idio_vol_pre": f"Usually moves about {daily(x)} a day on its own",
        "vol_pre": f"Usually moves about {daily(x)} a day",
        "yz_vol_5": "How much it swung in the week before the news",
        "yz_vol_20": "How much it swung in the month before the news",
        "yz_vol_60": "How much it swung in the three months before the news",
        "range_shock_01": "How wide its trading range was on the news",
        "rev_5": f"{'Rose' if up else 'Fell'} {pct(np.expm1(x))} in the week before the news",
        "max_20": f"Biggest one-day rise in the month before: {pct(x)}",
        "min_20": f"Biggest one-day fall in the month before: {pct(x)}",
        "mom_20": "Its trend over the month before the news",
        "mom_60": "Its trend over the three months before the news",
        "mom_250": "Its trend over the year before the news",
        "beta": f"Moves {times(x)} as much as the market",
        "log_mcap": "The company's size",
        "log_dollar_vol": "How much of it trades each day",
        "amihud": "How easily its shares trade",
        "exp_earn_in_win": "An earnings report is due in the next four weeks" if x > 0.5 else "No earnings report due in the next four weeks",
        "days_to_exp_earn": f"Next earnings report in about {int(x)} days",
        "days_since_earn": f"{int(x)} days since its last earnings report",
        "earn_react_hist": "How it reacted to past earnings reports",
        "n_events": "How much news came out at once",
        "clv_k": "Where it closed within today's range",
        "gap_k": "How it opened compared with the last close",
        "range_k_z": "How wide today's range was",
        "path_vol_ratio": "How choppy the moves since the news have been",
        "sr_dev_k": "Short selling today compared with normal",
        "sr3_dev": "Short selling over the last three days",
        "sr_day0_dev": "Short selling on the news day",
        "hy_oas": "Stress in the corporate bond market",
        "epu": "Economic policy uncertainty",
        "umcsent": "Consumer confidence",
        "skew_60": "How lopsided its recent moves have been",
        "ovn_share": "How much of the news move happened overnight",
        "pre_abvol_5": "Trading in the week before the news",
        "turn_chg": "How much its trading had picked up before the news",
        "alpha_pre": "How it did against the market in the year before the news",
        "mkt_pre5": "How the market moved in the week before the news",
        "mkt_vol20": "How jumpy the whole market was before the news",
        "scandal_word": "The headline points to a scandal" if x > 0.5 else "The headline does not point to a scandal",
        "yz_term": "Recent swings compared with its longer-run swings",
        "friday": "The news came out on a Friday" if x > 0.5 else "The news came out midweek",
        "amihud_chg": "How much easier or harder it has become to trade",
        "zero_ret_60": "How often the price did not move at all",
        "lv_long": "How much it usually swings",
        "lv_5": "How much it swung in the week before the news",
        "lv_20": "How much it swung in the month before the news",
        "lv_range": "How wide its daily ranges were before the news",
        "intra_01": "How much of the news move happened during trading hours",
        "ovn_01": "How much of the news move happened overnight",
        "seas_5y": "How it usually does at this time of year",
    }
    return E.get(f)


def explain(drivers, kind="other", quiet=False, top=4):
    """drivers = [(feature, contribution, value)] -> [{"text", "c"}] with plain names, duplicates merged."""
    out, seen = [], set()
    for f, c, x in drivers or []:
        if not np.isfinite(x):
            continue
        t = describe(f, x, kind, quiet)
        if not t or t in seen:
            continue
        seen.add(t); out.append({"text": t, "c": round(float(c), 4)})
        if len(out) == top:
            break
    return out
