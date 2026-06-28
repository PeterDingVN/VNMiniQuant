# THIS MODULE IS THE COPY OF git+https://github.com/rongardF/tvdatafeed.git@e6f6aaa7de439ac6e454d9b26d2760ded8dc4923
# CHANGE INCLUDE: 
# + iterative scrape and append data to exceed 5000 bars limit 
# Credit to: rongardF

import time
import datetime
import json
import logging
import random
import re
import string
import pandas as pd
from websocket import create_connection
import requests
import json

logger = logging.getLogger(__name__)
YELLOW = "\033[93m"
RESET = "\033[0m"


class TvSocket:
    __sign_in_url = 'https://www.tradingview.com/accounts/signin/'
    __ws_headers = json.dumps({"Origin": "https://data.tradingview.com"})
    __signin_headers = {'Referer': 'https://www.tradingview.com'}
    __ws_timeout = 5

    def __init__(
        self,
        username: str = None,
        password: str = None,
    ) -> None:

        self.ws_debug = False

        self.token = self.__auth(username, password)

        if self.token is None:
            self.token = "unauthorized_user_token"
            logger.warning(
                f"{YELLOW}[WARNING] You are in Trading View Guest Mode, data might be limited{RESET}"
            )

        self.ws = None
        self.session = self.__generate_session()
        self.chart_session = self.__generate_chart_session()

    def __auth(self, username, password):

        if (username is None or password is None):
            token = None

        else:
            data = {"username": username,
                    "password": password,
                    "remember": "on"}
            try:
                response = requests.post(
                    url=self.__sign_in_url, data=data, headers=self.__signin_headers)
                token = response.json()['user']['auth_token']
            except Exception:
                token = None

        return token

    def __create_connection(self):
        logging.debug("creating websocket connection")
        self.ws = create_connection(
            "wss://data.tradingview.com/socket.io/websocket", headers=self.__ws_headers, timeout=self.__ws_timeout
        )


    @staticmethod
    def __generate_session():
        stringLength = 12
        letters = string.ascii_lowercase
        random_string = "".join(random.choice(letters)
                                for i in range(stringLength))
        return "qs_" + random_string

    @staticmethod
    def __generate_chart_session():
        stringLength = 12
        letters = string.ascii_lowercase
        random_string = "".join(random.choice(letters)
                                for i in range(stringLength))
        return "cs_" + random_string

    @staticmethod
    def __prepend_header(st):
        return "~m~" + str(len(st)) + "~m~" + st

    @staticmethod
    def __construct_message(func, param_list):
        return json.dumps({"m": func, "p": param_list}, separators=(",", ":"))

    def __create_message(self, func, paramList):
        return self.__prepend_header(self.__construct_message(func, paramList))

    def __send_message(self, func, args):
        m = self.__create_message(func, args)
        if self.ws_debug:
            print(m)
        self.ws.send(m)


    @staticmethod
    def __create_df(raw_data, symbol):

        data = []
        messages = re.split(r'~m~\d+~m~', raw_data)

        for msg in messages:
            if not msg.strip():
                continue

            try:
                obj = json.loads(msg)
            except Exception:
                continue

            if obj.get("m") != "timescale_update":
                continue

            series = obj["p"][1]
            for _, series_data in series.items():

                if not isinstance(series_data, dict):
                    continue

                if "s" not in series_data:
                    continue

                for bar in series_data["s"]:

                    v = bar["v"]

                    ts = datetime.datetime.fromtimestamp(v[0], datetime.UTC)

                    volume = v[5] if len(v) > 5 else 0.0

                    data.append([
                        ts,
                        float(v[1]),
                        float(v[2]),
                        float(v[3]),
                        float(v[4]),
                        float(volume),
                    ])

        if not data:
            return None

        df = pd.DataFrame(
            data,
            columns=["datetime","open","high","low","close","volume"]).sort_values(by='datetime')

        return df

    @staticmethod
    def __format_symbol(symbol, exchange, contract: int = None):

        if ":" in symbol:
            pass
        elif contract is None:
            symbol = f"{exchange}:{symbol}"

        elif isinstance(contract, int):
            symbol = f"{exchange}:{symbol}{contract}!"

        else:
            raise ValueError("not a valid contract")

        return symbol

    def get_hist(
        self,
        symbol: str,
        exchange: str = "NSE",
        interval: str  = '1D',
        n_bars: int = 10,
        fut_contract: int = None,
        extended_session: bool = False,
        max_retries: int = 4,
        max_bars: int = 5000
    ) -> pd.DataFrame:
        
        
        symbol = self.__format_symbol(
            symbol=symbol, exchange=exchange, contract=fut_contract
        )

        for _ in range(max_retries):
            self.__create_connection()

            self.__send_message("set_auth_token", [self.token])
            self.__send_message("chart_create_session", [self.chart_session, ""])
            self.__send_message("quote_create_session", [self.session])
            self.__send_message(
                "quote_set_fields",
                [
                    self.session,
                    "ch",
                    "chp",
                    "current_session",
                    "description",
                    "local_description",
                    "language",
                    "exchange",
                    "fractional",
                    "is_tradable",
                    "lp",
                    "lp_time",
                    "minmov",
                    "minmove2",
                    "original_name",
                    "pricescale",
                    "pro_name",
                    "short_name",
                    "type",
                    "update_mode",
                    "volume",
                    "currency_code",
                    "rchp",
                    "rtc",
                ],
            )

            self.__send_message(
                "quote_add_symbols", [self.session, symbol,
                                    {"flags": ["force_permission"]}]
            )
            self.__send_message("quote_fast_symbols", [self.session, symbol])

            self.__send_message(
                "resolve_symbol",
                [
                    self.chart_session,
                    "symbol_1",
                    '={"symbol":"'
                    + symbol
                    + '","adjustment":"splits","session":'
                    + ('"regular"' if not extended_session else '"extended"')
                    + "}",
                ],
            )

            
            first_chunk = min(max_bars, n_bars)
            remaining = n_bars - first_chunk
            self.__send_message(
                "create_series",
                [self.chart_session, "s1", "s1", "symbol_1", interval, first_chunk],
            )

            self.__send_message("switch_timezone", [
                                self.chart_session, "Etc/UTC"])

            raw_data = ""

            logger.debug(f"getting data for {symbol}...")

            while True:
                try:
                    result = self.ws.recv()
                    raw_data += result + "\n"

                except Exception as e:
                    break

                # Ignore everything until the current batch finishes
                if "series_completed" not in result:
                    continue

                # Current batch finished
                if remaining <= 0:
                    break

                next_chunk = min(max_bars, remaining)
                remaining -= next_chunk

                logger.debug(
                    f"Requesting {next_chunk} more bars "
                    f"({remaining} remaining)"
                )

                self.__send_message(
                    "request_more_data",
                    [
                        self.chart_session,
                        "sds_1",      # verify this id is correct
                        next_chunk,
                    ],
                )

            time.sleep(0.3)

        return self.__create_df(raw_data, symbol)

