"""Multi-market trading bot starter.

Default behavior is dry-run. Set LIVE_TRADING=true only after testing each
adapter with demo accounts and understanding the broker's risk controls.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol


logging.basicConfig(
	level=os.getenv("LOG_LEVEL", "INFO").upper(),
	format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger("multi-market-bot")


def load_local_env() -> None:
	"""Load simple KEY=VALUE settings from a local .env file if present."""
	env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
	if not os.path.exists(env_path):
		return
	with open(env_path, encoding="utf-8") as env_file:
		for raw_line in env_file:
			line = raw_line.strip()
			if not line or line.startswith("#") or "=" not in line:
				continue
			key, value = line.split("=", 1)
			if key.strip() and key.strip() not in os.environ:
				os.environ[key.strip()] = value.strip().strip('"').strip("'")


load_local_env()


@dataclass(frozen=True)
class Settings:
	live_trading: bool = os.getenv("LIVE_TRADING", "false").lower() == "true"
	stock_symbols: tuple[str, ...] = tuple(
		symbol.strip().upper()
		for symbol in os.getenv("STOCK_SYMBOLS", "AAPL,MSFT").split(",")
		if symbol.strip()
	)
	mt5_symbols: tuple[str, ...] = tuple(
		symbol.strip().upper()
		for symbol in os.getenv("MT5_SYMBOLS", "EURUSD").split(",")
		if symbol.strip()
	)
	deriv_symbol: str = os.getenv("DERIV_SYMBOL", "R_100")
	deriv_product: str = os.getenv("DERIV_PRODUCT", "options").lower()
	deriv_contract: str = os.getenv("DERIV_CONTRACT", "rise_fall").lower()
	deriv_account_id: str = os.getenv("DERIV_ACCOUNT_ID", "")
	deriv_loginid: str = os.getenv("DERIV_LOGINID", "")
	deriv_currency: str = os.getenv("DERIV_CURRENCY", "USD").upper()
	fast_period: int = int(os.getenv("FAST_PERIOD", "9"))
	slow_period: int = int(os.getenv("SLOW_PERIOD", "21"))
	max_position_size: float = float(os.getenv("MAX_POSITION_SIZE", "100"))


@dataclass(frozen=True)
class Signal:
	symbol: str
	action: str
	price: float
	reason: str


class Broker(Protocol):
	def position(self, symbol: str) -> float: ...

	def submit(self, signal: Signal, quantity: float) -> None: ...


def ema(values: list[float], period: int) -> float:
	"""Return the latest exponential moving average."""
	if not values:
		raise ValueError("Cannot calculate EMA for empty data")
	multiplier = 2 / (period + 1)
	average = values[0]
	for value in values[1:]:
		average = (value - average) * multiplier + average
	return average


def crossover_signal(
	symbol: str,
	closes: list[float],
	fast_period: int,
	slow_period: int,
) -> Signal | None:
	"""Generate a signal only when the latest candle crosses the slow EMA."""
	if len(closes) < slow_period + 1:
		return None
	previous = closes[:-1]
	previous_fast = ema(previous[-slow_period:], fast_period)
	previous_slow = ema(previous[-slow_period:], slow_period)
	current_fast = ema(closes[-slow_period:], fast_period)
	current_slow = ema(closes[-slow_period:], slow_period)
	price = closes[-1]

	if previous_fast <= previous_slow and current_fast > current_slow:
		return Signal(symbol, "BUY", price, "fast EMA crossed above slow EMA")
	if previous_fast >= previous_slow and current_fast < current_slow:
		return Signal(symbol, "SELL", price, "fast EMA crossed below slow EMA")
	return None


class DryRunBroker:
	"""Broker used by default; it never sends an order to a third party."""

	def position(self, symbol: str) -> float:
		return 0.0

	def submit(self, signal: Signal, quantity: float) -> None:
		LOGGER.info("DRY RUN %s %s %.4f at %.5f (%s)", signal.action, signal.symbol, quantity, signal.price, signal.reason)


class AlpacaBroker:
	def __init__(self, settings: Settings) -> None:
		from alpaca.trading.client import TradingClient

		api_key = os.environ["ALPACA_API_KEY"]
		secret_key = os.environ["ALPACA_SECRET_KEY"]
		self.client = TradingClient(api_key, secret_key, paper=True)
		self.settings = settings

	def position(self, symbol: str) -> float:
		try:
			return float(self.client.get_open_position(symbol).qty)
		except Exception:
			return 0.0

	def submit(self, signal: Signal, quantity: float) -> None:
		from alpaca.trading.enums import OrderSide, TimeInForce
		from alpaca.trading.requests import MarketOrderRequest

		order = MarketOrderRequest(
			symbol=signal.symbol,
			qty=quantity,
			side=OrderSide.BUY if signal.action == "BUY" else OrderSide.SELL,
			time_in_force=TimeInForce.DAY,
		)
		self.client.submit_order(order_data=order)
		LOGGER.info("Submitted Alpaca paper order: %s %s %.4f", signal.action, signal.symbol, quantity)


class MT5Broker:
	def __init__(self, settings: Settings) -> None:
		import MetaTrader5 as mt5

		self.mt5 = mt5
		self.settings = settings
		if not mt5.initialize():
			raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

	def position(self, symbol: str) -> float:
		positions = self.mt5.positions_get(symbol=symbol) or ()
		return sum(float(position.volume) for position in positions)

	def submit(self, signal: Signal, quantity: float) -> None:
		tick = self.mt5.symbol_info_tick(signal.symbol)
		if tick is None:
			raise RuntimeError(f"No MT5 tick available for {signal.symbol}")
		order_type = self.mt5.ORDER_TYPE_BUY if signal.action == "BUY" else self.mt5.ORDER_TYPE_SELL
		price = tick.ask if signal.action == "BUY" else tick.bid
		request: dict[str, Any] = {
			"action": self.mt5.TRADE_ACTION_DEAL,
			"symbol": signal.symbol,
			"volume": quantity,
			"type": order_type,
			"price": price,
			"deviation": 20,
			"magic": 260903,
			"comment": "ema-cross-bot",
			"type_time": self.mt5.ORDER_TIME_GTC,
			"type_filling": self.mt5.ORDER_FILLING_IOC,
		}
		result = self.mt5.order_send(request)
		if result is None or result.retcode != self.mt5.TRADE_RETCODE_DONE:
			raise RuntimeError(f"MT5 order failed: {result}")
		LOGGER.info("Submitted MT5 order: %s %s %.4f", signal.action, signal.symbol, quantity)


class DerivBroker:
	def __init__(self, settings: Settings) -> None:
		import websocket

		self.websocket = websocket
		self.settings = settings
		self.app_id = os.getenv("DERIV_APP_ID", "1089")
		self.token = os.environ.get("DERIV_TOKEN", "")
		self.socket: Any = None
		self.market_socket: Any = None
		if not self.app_id.isdigit() or self.app_id in {"1089", "0"}:
			raise RuntimeError(
				"DERIV_APP_ID must be the numeric App ID linked to this token; "
				"replace the placeholder in C:\\Users\\Jay\\.env."
			)
		if not settings.deriv_account_id.isdigit() or settings.deriv_account_id.lower().startswith("your_"):
			raise RuntimeError(
				"DERIV_ACCOUNT_ID must be the numeric Deriv Options account ID; "
				"replace the placeholder in C:\\Users\\Jay\\.env."
			)
		if settings.deriv_product != "options":
			raise ValueError("DERIV_PRODUCT must be 'options'")
		if settings.deriv_contract != "rise_fall":
			raise ValueError("DERIV_CONTRACT must be 'rise_fall'")
		if not settings.deriv_account_id:
			raise RuntimeError("DERIV_ACCOUNT_ID is required for the Deriv options API")
		if not self.token or self.token.lower() in {
			"your_token_here",
			"your_actual_new_token",
			"replace_with_a_new_deriv_api_token",
		}:
			raise RuntimeError(
				"DERIV_TOKEN is missing or still a placeholder. "
				"Create a new Deriv API token and put it in C:\\Users\\Jay\\.env."
			)

	def connect(self) -> None:
		otp_url = (
			"https://api.derivws.com/trading/v1/options/accounts/"
			f"{self.settings.deriv_account_id}/otp"
		)
		request = Request(
			otp_url,
			method="POST",
			headers={
				"Deriv-App-ID": self.app_id,
				"Authorization": f"Bearer {self.token}",
				"Content-Type": "application/json",
			},
			data=b"{}",
		)
		try:
			with urlopen(request, timeout=15) as response:
				otp_response = json.loads(response.read().decode("utf-8"))
		except HTTPError as error:
			detail = error.read().decode("utf-8", errors="replace")
			try:
				error_data = json.loads(detail)
				detail = "; ".join(
					f"{item.get('code', 'Error')}: {item.get('message', 'Unknown error')}"
					for item in error_data.get("errors", [])
				) or detail
			except json.JSONDecodeError:
				pass
			raise RuntimeError(f"Deriv OTP request failed ({error.code}): {detail}") from error
		except URLError as error:
			raise RuntimeError(f"Could not reach Deriv OTP endpoint: {error.reason}") from error

		websocket_url = otp_response.get("websocket_url") or otp_response.get("data", {}).get("websocket_url")
		if not websocket_url:
			raise RuntimeError(f"Deriv OTP response did not contain websocket_url: {otp_response}")
		self.socket = self.websocket.create_connection(websocket_url, timeout=15)
		LOGGER.info("Connected to Deriv options trading WebSocket for account %s", self.settings.deriv_account_id)

	def connect_market_data(self) -> None:
		url = f"wss://ws.derivws.com/websockets/v3?app_id={self.app_id}"
		self.market_socket = self.websocket.create_connection(url, timeout=15)

	def request(self, payload: dict[str, Any]) -> dict[str, Any]:
		if self.socket is None:
			raise RuntimeError("Deriv socket is not connected")
		self.socket.send(json.dumps(payload))
		while True:
			response = json.loads(self.socket.recv())
			if "error" in response:
				raise RuntimeError(f"Deriv API error: {response['error']}")
			if "msg_type" in response or "authorize" in response or "proposal" in response or "buy" in response:
				return response

	def ticks(self, symbol: str, count: int) -> list[float]:
		if self.market_socket is None:
			self.connect_market_data()
		self.market_socket.send(json.dumps({"ticks_history": symbol, "count": count, "end": "latest", "style": "ticks"}))
		response = json.loads(self.market_socket.recv())
		if "error" in response:
			raise RuntimeError(f"Deriv market-data error: {response['error']}")
		prices = response.get("history", {}).get("prices", [])
		if len(prices) < count:
			raise RuntimeError(f"Deriv returned only {len(prices)} ticks for {symbol}")
		return [float(price) for price in prices]

	def close(self) -> None:
		if self.socket is not None:
			self.socket.close()
			self.socket = None
		if self.market_socket is not None:
			self.market_socket.close()
			self.market_socket = None

	def position(self, symbol: str) -> float:
		return 0.0

	def check_connection(self) -> None:
		try:
			self.connect()
		finally:
			self.close()

	def submit(self, signal: Signal, quantity: float) -> None:
		contract_type = "RISE" if signal.action == "BUY" else "FALL"
		proposal = {
			"proposal": 1,
			"amount": round(quantity, 2),
			"basis": "stake",
			"contract_type": contract_type,
			"currency": self.settings.deriv_currency,
			"duration": int(os.getenv("DERIV_DURATION", "5")),
			"duration_unit": os.getenv("DERIV_DURATION_UNIT", "t"),
			"symbol": signal.symbol,
		}
		if self.socket is None:
			self.connect()
		response = self.request(proposal)
		proposal_data = response["proposal"]
		buy_price = float(proposal_data["ask_price"])
		result = self.request({"buy": proposal_data["id"], "price": buy_price})
		if "buy" not in result:
			raise RuntimeError("Deriv buy response did not contain a contract")
		LOGGER.info("Submitted Deriv Rise/Fall %s contract on %s", contract_type, signal.symbol)


def load_stock_closes(symbol: str) -> list[float]:
	import yfinance as yf

	data = yf.download(symbol, period="3mo", interval="1h", progress=False, auto_adjust=False)
	if data.empty:
		raise RuntimeError(f"No stock data returned for {symbol}")
	close = data["Close"]
	if hasattr(close, "iloc") and getattr(close, "ndim", 1) > 1:
		close = close.iloc[:, 0]
	return [float(value) for value in close.dropna().tolist()]


def choose_broker(settings: Settings, market: str) -> Broker:
	if not settings.live_trading:
		return DryRunBroker()
	if market == "stocks":
		return AlpacaBroker(settings)
	if market == "mt5":
		return MT5Broker(settings)
	if market == "deriv":
		return DerivBroker(settings)
	raise ValueError(f"Unsupported market: {market}")


def run_stocks(settings: Settings) -> None:
	broker = choose_broker(settings, "stocks")
	for symbol in settings.stock_symbols:
		closes = load_stock_closes(symbol)
		signal = crossover_signal(symbol, closes, settings.fast_period, settings.slow_period)
		if signal is None:
			LOGGER.info("No stock signal for %s", symbol)
			continue
		quantity = min(settings.max_position_size / signal.price, settings.max_position_size)
		if signal.action == "SELL" and broker.position(symbol) <= 0:
			LOGGER.info("Ignoring sell signal for %s because no long position exists", symbol)
			continue
		broker.submit(signal, round(quantity, 4))


def run_deriv_demo(settings: Settings) -> None:
	"""Read Deriv volatility ticks and optionally submit one contract."""
	broker = DerivBroker(settings)
	broker.connect()
	try:
		prices = broker.ticks(settings.deriv_symbol, settings.slow_period + 2)
		LOGGER.info("Read %s authenticated Deriv ticks", settings.deriv_symbol)
		signal = crossover_signal(settings.deriv_symbol, prices, settings.fast_period, settings.slow_period)
		if signal and settings.live_trading:
			broker.submit(signal, float(os.getenv("DERIV_STAKE", "1")))
		elif signal:
			LOGGER.info("DERIV OPTIONS DRY-RUN SIGNAL %s", signal)
	finally:
		broker.close()


def run_mt5(settings: Settings) -> None:
	import MetaTrader5 as mt5

	broker = choose_broker(settings, "mt5")
	timeframe = getattr(mt5, os.getenv("MT5_TIMEFRAME", "TIMEFRAME_M5"), mt5.TIMEFRAME_M5)
	for symbol in settings.mt5_symbols:
		if not mt5.symbol_select(symbol, True):
			LOGGER.warning("MT5 symbol is unavailable: %s", symbol)
			continue
		rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, settings.slow_period + 2)
		if rates is None or len(rates) < settings.slow_period + 1:
			LOGGER.warning("Not enough MT5 candles for %s", symbol)
			continue
		closes = [float(rate["close"]) for rate in rates]
		signal = crossover_signal(symbol, closes, settings.fast_period, settings.slow_period)
		if signal:
			quantity = min(settings.max_position_size / signal.price, settings.max_position_size)
			broker.submit(signal, round(quantity, 2))


def list_mt5_symbols() -> None:
	import MetaTrader5 as mt5

	if not mt5.initialize():
		raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
	try:
		query = os.getenv("MT5_SYMBOL_SEARCH", "EUR").upper()
		symbols = mt5.symbols_get() or ()
		matches = [symbol.name for symbol in symbols if query in symbol.name.upper()]
		if matches:
			print("\n".join(matches))
		else:
			print(f"No MT5 symbols matched {query!r}")
	finally:
		mt5.shutdown()


def main() -> None:
	parser = argparse.ArgumentParser(description="Deriv Rise/Fall options, stocks, and MT5 trading bot")
	parser.add_argument("--market", choices=("stocks", "deriv", "mt5"), default="deriv")
	parser.add_argument("--once", action="store_true", help="Run one scan and exit")
	parser.add_argument("--check-deriv", action="store_true", help="Authorize with Deriv without requesting or buying a contract")
	parser.add_argument("--list-mt5-symbols", action="store_true", help="List MT5 symbols matching MT5_SYMBOL_SEARCH")
	args = parser.parse_args()
	settings = Settings()
	LOGGER.info("Starting in %s mode at %s", "LIVE" if settings.live_trading else "DRY RUN", datetime.now(timezone.utc).isoformat())
	if args.check_deriv:
		DerivBroker(settings).check_connection()
		return
	if args.list_mt5_symbols:
		list_mt5_symbols()
		return

	while True:
		if args.market == "stocks":
			run_stocks(settings)
		elif args.market == "deriv":
			run_deriv_demo(settings)
		else:
			run_mt5(settings)
		if args.once:
			return
		time.sleep(3600)


if __name__ == "__main__":
	main()
