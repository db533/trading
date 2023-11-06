from candlestick.patterns.candlestick_finder import CandlestickFinder

class BullishKicker(CandlestickFinder):
    def __init__(self, target=None):
        super().__init__(self.get_class_name(), 3, target=target)

    def logic(self, idx):
        candle = self.data.iloc[idx]
        prev_candle = self.data.iloc[idx + 1 * self.multi_coeff]
        b_prev_candle = self.data.iloc[idx + 2 * self.multi_coeff]

        close = candle[self.close_column]
        open = candle[self.open_column]

        prev_close = prev_candle[self.close_column]
        prev_open = prev_candle[self.open_column]

        return (
               prev_close < prev_open  # Prior candle is downward
               and (open >= prev_open)  # Next candle opens above the prior candle's open.
               and (close > open ) # Next candle closes above
        )
