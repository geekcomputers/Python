import ccxt


def getprice(symbol, exchange_id):
    """
    :param symbol: The cryptocurrency symbol.
    :type symbol: str
        :param exchange_id: The exchange ID.
        :type exchange_id: str

        :returns v_price
    -- the price of the cryptocurrency in USD or BTC, depending on whether it is a fiat currency or not (float)
    """
    symbol = symbol.upper()  # BTC/USDT, LTC/USDT, ETH/BTC, LTC/BTC
    exchange_id = exchange_id.lower()  # binance, #bitmex
    symbol_1 = symbol.split("/")
    exchange = getattr(ccxt, exchange_id)({
        # https://github.com/ccxt/ccxt/wiki/Manual#rate-limit
        'enableRateLimit': True
    })
    try:
        v_price = exchange.fetch_ticker(symbol)
        r_price = v_price['info']['lastPrice']
        if (symbol_1[1] == "USD" or symbol_1[1] == "USDT"):
            v_return = "{:.2f} {}".format(float(r_price), symbol_1[1])
            return v_return
        else:
            v_return = "{:.8f} {}".format(float(r_price), symbol_1[1])
            return v_return
    except (ccxt.ExchangeError, ccxt.NetworkError) as error:
        # add necessary handling or rethrow the exception
        return 'Got an error', type(error).__name__, error.args
    raise


print(getprice("btc/usdt", "BINANCE"))
print(getprice("btc/usd", "BITMEX"))
