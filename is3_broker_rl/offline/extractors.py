from os import PathLike
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import pandas as pd
from pandas.errors import EmptyDataError


def pd_glob_read_csv(
    search_path: Union[str, bytes, PathLike],
    glob: str,
    sep: str = ",",
    index_col: Union[bool, int] = False,
    skiprows: Optional[int] = None,
    skipinitialspace: bool = True,
    predicate: Optional[Callable[[pd.DataFrame], Sequence[bool]]] = None,
    header: Optional[int] = 0,
    names: Optional[List[str]] = None,
    usecols: Optional[List[int]] = None,
) -> List[pd.DataFrame]:
    search_path = Path(search_path)
    if not search_path.exists():
        raise FileNotFoundError(f"The path {search_path} cannot be found.")

    search_result = list(search_path.glob(glob))
    if not search_result:
        raise FileNotFoundError(f"Cannot find files for glob pattern {glob} within {search_path}.")

    dataframes: List[pd.DataFrame] = []

    for i, file_path in enumerate(search_result):
        # Try to extract the game_id from the filename
        if len(file_path.name.split(".")) > 1:
            game_id = file_path.name.split(".")[0]
        else:
            game_id = i

        try:
            df = pd.read_csv(
                file_path,
                sep=sep,
                index_col=index_col,
                skiprows=skiprows,
                skipinitialspace=skipinitialspace,
                header=header,
                names=names,
                usecols=usecols,
            )
            if len(df) > 0:
                # Append game_id from filename to dataframe
                df["game_id"] = game_id
                if predicate is not None:
                    df = df[predicate(df)]
                dataframes.append(df)
        except EmptyDataError:
            continue

    return dataframes
