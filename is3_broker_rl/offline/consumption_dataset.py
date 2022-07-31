import re
from concurrent import futures
from functools import reduce
from io import StringIO
from time import time
from typing import List

import numpy as np
import pandas as pd

from is3_broker_rl.api.dto import MarketPosition
from is3_broker_rl.offline.extractors import pd_glob_read_csv

BASE_PATH = "/Users/phipag/Git/powertac/is3-powertac-analysis/files/finals_2021"


def parse_imbalance_stats_df(broker_name: str) -> pd.DataFrame:
    raw_dfs = pd_glob_read_csv(search_path=BASE_PATH, glob="**/*imbalance-stats.csv")
    imbalance_df = pd.concat(raw_dfs, axis=0)
    imbalance_df = (
        imbalance_df[["game_id", "timeslot", broker_name, "imbalance"]]
        .dropna()
        .rename(columns={broker_name: "ownImbalanceKwh", "imbalance": "gridImbalance"})
        .set_index(["game_id", "timeslot"], drop=True)
    )
    return imbalance_df


def parse_tariff_market_share_df(broker_name: str) -> pd.DataFrame:
    raw_dfs = pd_glob_read_csv(search_path=BASE_PATH, glob="**/*tariff-market-share.csv", sep=";")

    preprocessed_dfs: List[pd.DataFrame] = []
    for df in raw_dfs:
        # Filter for our own consumption.
        df_consumption_own_broker = (
            df[["game_id", "timeslot", "broker", "type", "subscriptions"]]
            .where(df["broker"] == broker_name)
            .where(df["type"] == "CONSUMPTION")
            .dropna()
            .astype({"timeslot": "int32", "subscriptions": "int32"})
            .groupby(["game_id", "timeslot"])
            .sum()
            .rename(columns={"subscriptions": "customerCount"})
        )
        if df_consumption_own_broker.empty:
            # Continue if we did not participate in the game.
            continue
        # Filter for the consumption of the other brokers.
        df_consumption_other_brokers = (
            df[["game_id", "timeslot", "broker", "type", "subscriptions"]]
            .where(df["broker"] != broker_name)
            .where(df["type"] == "CONSUMPTION")
            .dropna()
            .astype({"timeslot": "int32", "subscriptions": "int32"})
            .groupby(["game_id", "timeslot"])
            .sum()
            .rename(columns={"subscriptions": "customerCount"})
        )
        # Calculate our consumption share.
        df_consumption_own_broker["consumptionShare"] = df_consumption_own_broker["customerCount"] / (
            # 1e-8 is to avoid division by zero if there is no consumption in a timeslot.
            df_consumption_own_broker["customerCount"]
            + df_consumption_other_brokers["customerCount"]
            + 1e-8
        )
        # Filter for our own production.
        df_production_own_broker = (
            df[["game_id", "timeslot", "broker", "type", "subscriptions"]]
            .where(df["broker"] == broker_name)
            .where(
                (df["type"] == "PRODUCTION") | (df["type"] == "WIND_PRODUCTION") | (df["type"] == "SOLAR_PRODUCTION")
            )
            .dropna()
            .astype({"timeslot": "int32", "subscriptions": "int32"})
            .groupby(["game_id", "timeslot"])
            .sum()
            .rename(columns={"subscriptions": "customerCount"})
        )
        # Filter for the production of the other brokers.
        df_production_other_brokers = (
            df[["game_id", "timeslot", "broker", "type", "subscriptions"]]
            .where(df["broker"] != broker_name)
            .where(
                (df["type"] == "PRODUCTION") | (df["type"] == "WIND_PRODUCTION") | (df["type"] == "SOLAR_PRODUCTION")
            )
            .dropna()
            .astype({"timeslot": "int32", "subscriptions": "int32"})
            .groupby(["game_id", "timeslot"])
            .sum()
            .rename(columns={"subscriptions": "customerCount"})
        )
        # Calculate our own production share.
        df_consumption_own_broker["productionShare"] = df_production_own_broker["customerCount"] / (
            # 1e-8 is to avoid division by zero if there is no consumption in a timeslot.
            df_production_own_broker["customerCount"]
            + df_production_other_brokers["customerCount"]
            + 1e-8
        )
        # When there are NaN values it means that our broker had not tariff active at that timeslot which is equal to a
        # 0% share.
        df_consumption_own_broker[["consumptionShare", "productionShare"]] = df_consumption_own_broker[
            ["consumptionShare", "productionShare"]
        ].fillna(0.0)
        df_consumption_own_broker = df_consumption_own_broker.reset_index()

        # The customerCount is only reported every 6 (or sometimes less) timeslots. We create this mask to determine
        # how often to repeat the rows to fill the timeslots in between.
        repeat_mask = df_consumption_own_broker.index.repeat(
            df_consumption_own_broker["timeslot"]
            .diff()
            .fillna(df_consumption_own_broker["timeslot"].iloc[1] - df_consumption_own_broker["timeslot"].iloc[0])
        )
        df_consumption_own_broker = df_consumption_own_broker.loc[repeat_mask].reset_index(drop=True)
        # We do not want to repeat the last timeslot and drop it again here.
        df_consumption_own_broker = df_consumption_own_broker.drop(df_consumption_own_broker.tail(5).index, axis=0)
        # Now, we still need to make the timeslot strictly increasing by one.
        first_timeslot = df_consumption_own_broker.iloc[0, 1]
        last_timeslot = df_consumption_own_broker.iloc[-1, 1]
        df_consumption_own_broker["timeslot"] = range(first_timeslot, last_timeslot + 1)
        preprocessed_dfs.append(df_consumption_own_broker)

    return pd.concat(preprocessed_dfs, axis=0).set_index(["game_id", "timeslot"], drop=True)


def parse_broker_accounting_df(broker_name: str) -> pd.DataFrame:
    raw_dfs = pd_glob_read_csv(search_path=BASE_PATH, glob="**/*broker-accounting.csv")
    filtered_dfs = [df for df in raw_dfs if any(df[col].iloc[0] == broker_name for col in df.columns)]

    def find_broker_column_index_for_name(df: pd.DataFrame, broker_name: str) -> int:
        for column in df.columns:
            if df.loc[0, column] == broker_name:
                return int(column[-1])
        return -1

    preprocessed_dfs: List[pd.DataFrame] = []
    for df in filtered_dfs:
        broker_column_index = find_broker_column_index_for_name(df, broker_name)
        df = df[
            [
                "game_id",
                "ts",
                f"cash.{broker_column_index}",
                f"ctx-c.{broker_column_index}",
                f"ctx-d.{broker_column_index}",
                f"mtx-c.{broker_column_index}",
                f"mtx-d.{broker_column_index}",
            ]
        ].rename(columns={"ts": "timeslot", f"cash.{broker_column_index}": "cashPosition"})
        df["capacity_costs"] = df[f"ctx-c.{broker_column_index}"] + df[f"ctx-d.{broker_column_index}"]
        df["wholesale_costs"] = df[f"mtx-c.{broker_column_index}"] + df[f"mtx-d.{broker_column_index}"]
        df = df.drop(
            columns=[
                f"ctx-c.{broker_column_index}",
                f"ctx-d.{broker_column_index}",
                f"mtx-c.{broker_column_index}",
                f"mtx-d.{broker_column_index}",
            ]
        )
        preprocessed_dfs.append(df)

    broker_accounting_df = pd.concat(preprocessed_dfs, axis=0).set_index(["game_id", "timeslot"], drop=True)

    return broker_accounting_df


def parse_production_consumption_df(broker_name: str) -> pd.DataFrame:
    raw_dfs = pd_glob_read_csv(
        search_path=BASE_PATH,
        glob="**/*production-consumption-transactions.csv",
        sep=";",
        predicate=lambda df: (df["tariff-power-type"] == "CONSUMPTION"),
    )
    consumption_df = pd.concat(raw_dfs).set_index(["game_id", "timeslot"], drop=True)
    own_consumption_df = consumption_df[consumption_df["tariff-broker"] == broker_name].copy()
    own_consumption_df["kwh_price"] = own_consumption_df["ttx-charge"] / own_consumption_df["ttx-kwh"]
    others_consumption_df = consumption_df[consumption_df["tariff-broker"] != broker_name].copy()
    others_consumption_df["kwh_price"] = others_consumption_df["ttx-charge"] / others_consumption_df["ttx-kwh"]

    kwh_price_df = pd.DataFrame(columns=["max_competitor_rate", "min_competitor_rate", "own_rate"])
    kwh_price_df["max_competitor_rate"] = others_consumption_df[["kwh_price"]].groupby(["game_id", "timeslot"]).max()
    kwh_price_df["min_competitor_rate"] = others_consumption_df[["kwh_price"]].groupby(["game_id", "timeslot"]).min()
    kwh_price_df["own_rate"] = own_consumption_df[["kwh_price"]].groupby(["game_id", "timeslot"]).mean()
    kwh_price_df["customerNetDemand"] = own_consumption_df[["ttx-kwh"]].groupby(["game_id", "timeslot"]).sum()

    market_leader_mask = kwh_price_df["own_rate"] > kwh_price_df["max_competitor_rate"]
    market_trailer_mask = kwh_price_df["own_rate"] < kwh_price_df["min_competitor_rate"]
    market_average_mask = (kwh_price_df["own_rate"] > kwh_price_df["min_competitor_rate"]) & (
        kwh_price_df["own_rate"] < kwh_price_df["max_competitor_rate"]
    )

    result_df = kwh_price_df[["customerNetDemand"]].copy()
    result_df.loc[market_leader_mask, "marketPosition"] = MarketPosition.LEADER.value
    result_df.loc[market_trailer_mask, "marketPosition"] = MarketPosition.TRAILER.value
    result_df.loc[market_average_mask, "marketPosition"] = MarketPosition.AVERAGE.value
    result_df["customerNetDemand"] = result_df["customerNetDemand"].fillna(0)
    result_df["marketPosition"] = result_df["marketPosition"].fillna(MarketPosition.NONE.value)
    result_df["consumption_profit"] = own_consumption_df.groupby(["game_id", "timeslot"])["ttx-charge"].sum()
    result_df["consumption_profit"] = result_df["consumption_profit"].fillna(0.0)

    return result_df


def parse_market_price_stats_df() -> pd.DataFrame:
    raw_dfs = pd_glob_read_csv(
        search_path=BASE_PATH,
        glob="**/*market-price-stats.csv",
        sep=",",
        header=None,
        names=["timeslot", *[f"mwh_price.{i}" for i in range(24)]],
        # Column 0 is timeslot and the other 24 are the clearing prices on the wholesale market.
        usecols=[0, *range(3, 24 + 3)],
    )
    market_price_df = pd.concat(raw_dfs).set_index(["game_id", "timeslot"], drop=True)
    mwh_tuple_regex = re.compile(r"\[(-?\d+\.?\d*)\s(-?\d+\.?\d*)]")
    for col in market_price_df.columns:
        if col.startswith("mwh_price"):
            market_price_df[col] = (
                market_price_df[col]
                # The second entry of the [mwh price] tuple is the wholesale clearing price.
                .apply(lambda value: mwh_tuple_regex.match(value).groups()[1]).astype("float32")
            )
    # We want to ignore zero values for calculation of the mean wholesale price and therefore set them to nan.
    market_price_df = market_price_df.replace(0.0, np.nan)
    market_price_df["wholesalePrice"] = market_price_df.mean(axis=1)

    return market_price_df[["wholesalePrice"]]


def parse_broker_market_prices_df(broker_name: str) -> pd.DataFrame:
    raw_dfs = pd_glob_read_csv(
        search_path=BASE_PATH,
        glob="**/*broker-market-prices.csv",
        # This CSV file is invalid because of the comma in the mwh_price tuples. Therefore, we parse it as a plain
        # string per row and escape the comma manually.
        sep="?",
    )

    mwh_tuple_regex = re.compile(r"(\[-?\d+\.?\d*)(,)(\s-?\d+\.?\d*])")
    preprocessed_dfs: List[pd.DataFrame] = []
    for invalid_df in raw_dfs:
        # Omit and save the only valid column "game_id" for later.
        game_id = invalid_df["game_id"]
        invalid_df = invalid_df.drop(columns="game_id")

        # Escape the comma in the string values of mwh_price tuples manually using regex.
        broker_market_price_string = mwh_tuple_regex.sub(
            "\g<1>\\\\\g<2>\\\\\g<3>", invalid_df.to_string(index=False)  # noqa: W605
        )
        # Read the dataframe again from the escaped string.
        invalid_df = pd.read_csv(StringIO(broker_market_price_string), sep=",", escapechar="\\")
        invalid_df.columns = [col.strip() for col in invalid_df.columns]
        # Append the game_id again.
        invalid_df["game_id"] = game_id

        preprocessed_dfs.append(invalid_df)

    broker_market_price_df = pd.concat(preprocessed_dfs)
    # Replace the mwh_price tuple by the ownWholesalePrice value only
    broker_market_price_df = (
        broker_market_price_df[["game_id", "ts", broker_name]]
        # Games where the broker did not participate yield nan values.
        .dropna(axis=0)
        .rename(columns={"ts": "timeslot", broker_name: "ownWholesalePrice"})
        .set_index(["game_id", "timeslot"], drop=True)
    )
    broker_market_price_df["ownWholesalePrice"] = (
        broker_market_price_df["ownWholesalePrice"]
        .str.strip()
        # The second entry of the [mwh, price] tuple is the own wholesale price.
        .apply(lambda value: re.match(r"\[(-?\d+\.?\d*),\s(-?\d+\.?\d*)]", value).groups()[1])
        .astype("float32")
        .abs()
    )

    return broker_market_price_df


def parse_balancing_transactions_df(broker_name: str) -> pd.DataFrame:
    raw_dfs = pd_glob_read_csv(
        search_path=BASE_PATH,
        glob="**/*balancing-market-transactions.csv",
        sep=";",
        predicate=lambda df: df["broker-name"] == broker_name,
    )
    balancing_df = pd.concat(raw_dfs, axis=0)
    balancing_df = balancing_df.set_index(["game_id", "timeslot"], drop=True).rename(
        columns={"charge": "balancing_costs"}
    )

    return balancing_df[["balancing_costs"]]


def main():
    start = time()
    with futures.ProcessPoolExecutor() as executor:
        broker_accounting_df = executor.submit(parse_broker_accounting_df, "TUC_TAC")
        imbalance_df = executor.submit(parse_imbalance_stats_df, "TUC_TAC")
        tariff_mkt_share_df = executor.submit(parse_tariff_market_share_df, "TUC_TAC")
        production_consumption_df = executor.submit(parse_production_consumption_df, "TUC_TAC")
        market_price_df = executor.submit(parse_market_price_stats_df)
        broker_market_prices_df = executor.submit(parse_broker_market_prices_df, "TUC_TAC")
        balancing_transactions_df = executor.submit(parse_balancing_transactions_df, "TUC_TAC")

    observation_df = reduce(
        lambda left_df, right_df: pd.merge(left_df, right_df, how="inner", left_index=True, right_index=True),
        [
            broker_accounting_df.result(),
            imbalance_df.result(),
            tariff_mkt_share_df.result(),
            production_consumption_df.result(),
            market_price_df.result(),
            broker_market_prices_df.result(),
            balancing_transactions_df.result(),
        ],
    )
    observation_df["reward"] = (
        observation_df["consumption_profit"]
        + observation_df["capacity_costs"]
        + observation_df["wholesale_costs"]
        + observation_df["balancing_costs"]
    )
    print(f"Execution duration: {time() - start:2f}s")
    print(observation_df)
    observation_df.to_csv("/Users/phipag/Git/powertac/is3-broker-rl/data/tuc_tac_offline.csv")


if __name__ == "__main__":
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.max_rows", 20)
    pd.set_option("display.width", 1000)
    main()
