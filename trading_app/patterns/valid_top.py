from candlestick.patterns.candlestick_finder import CandlestickFinder

class ThreeWhiteSoldiers(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 3, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]
        b_prev_candle = self.data.iloc[idx + 2 * self.multi_coeff]
        b2_prev_candle = self.data.iloc[idx + 3 * self.multi_coeff]
        b3_prev_candle = self.data.iloc[idx + 4 * self.multi_coeff]

        close = candle[self.close_column]
        open = candle[self.open_column]
        high = candle[self.high_column]
        low = candle[self.low_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]
        prev_high = prev_candle[self.high_column]
        prev_low = prev_candle[self.low_column]

        b_prev_close = b_prev_candle[self.close_column]
        b_prev_open = b_prev_candle[self.open_column]
        b_prev_high = b_prev_candle[self.high_column]
        b_prev_low = b_prev_candle[self.low_column]

        b2_prev_close = b2_prev_candle[self.close_column]
        b2_prev_open = b2_prev_candle[self.open_column]
        b2_prev_high = b2_prev_candle[self.high_column]
        b2_prev_low = b2_prev_candle[self.low_column]

        b3_prev_close = b3_prev_candle[self.close_column]
        b3_prev_open = b3_prev_candle[self.open_column]
        b3_prev_high = b3_prev_candle[self.high_column]
        b3_prev_low = b3_prev_candle[self.low_column]



        return (
            b_prev_close < b_prev_open
            and abs(b_prev_close - b_prev_open) / (b_prev_high - b_prev_low) >= 0.7
            and abs(prev_close - prev_open) / (prev_high - prev_low) < 0.1
            and close > open
            and abs(close - open) / (high - low) >= 0.7
            and b_prev_close > prev_close
            and b_prev_close > prev_open
            and prev_close < open
            and prev_open < open
            and close > b_prev_close
            and (prev_high - max(prev_close, prev_open)) > (3 * abs(prev_close - prev_open))
            and (min(prev_close, prev_open) - prev_low) > (3 * abs(prev_close - prev_open))
        )
