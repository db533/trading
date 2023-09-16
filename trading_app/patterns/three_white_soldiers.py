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

        b_body_long = ((b_prev_high - b_prev_low) * .95 > (b_prev_open - b_prev_close))
        prev_body_long = ((prev_high - prev_low) * .95 > (prev_open - prev_close))
        body_long = ((high - low) * .95 > (open - close))

        return (
               b2_prev_close < b2_prev_open  # Oldest candle is downward
               and (b_body_long and b_prev_close > b_prev_open)  # Next candle has large body and closes higher
               and (prev_body_long and prev_open > b_prev_open and prev_close > b_prev_close) # Next candle has large body, opens above prior open and below or equal to prior clode and closes higher
               and (body_long and open > prev_open and close > prev_close) # Last candle has large body, opens above prior open and below or equal to prior clode and closes higher
        )
