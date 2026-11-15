#!/usr/bin/env Rscript
# ==============================================================================
# Sentinel-Cap | Sprint 2: R Time-Series Refinery & Econometric Priors
# Implements Section 3.4 (Eq. 5) and Section 4.1 (Eq. 6, 7, 8, 9)
# ==============================================================================

suppressPackageStartupMessages({
  library(arrow)
  library(xts)
  library(zoo)
})

cat("[R Refinery 1/3] Loading Market Panel & 365-Day Calendar Event Matrix...\n")
mkt_df <- read_parquet("data/02_interim/market_macro_panel.parquet")
evt_df <- read_parquet("data/01_raw/gdelt_global_calendar_events.parquet")

mkt_df$timestamp <- as.Date(mkt_df$timestamp)
evt_df$timestamp <- as.Date(evt_df$timestamp)

# Extract discrete NSE trading days vs 365-day continuous calendar grid
trading_days <- sort(unique(mkt_df$timestamp))
event_cols <- grep("^E_", colnames(evt_df), value = TRUE)

cat("[R Refinery 2/3] Applying Equation (5): Weekend/Holiday Accumulation & zoo::na.locf...\n")
# Map every calendar day tau (Fri post-close, Sat, Sun, holidays) to the NEXT valid NSE trading day
next_trading_idx <- findInterval(evt_df$timestamp - 1, trading_days) + 1
valid_mask <- next_trading_idx <= length(trading_days)
evt_df <- evt_df[valid_mask, ]
evt_df$mapped_trading_day <- trading_days[next_trading_idx[valid_mask]]

# Sum cumulative event impulses over non-trading days into Monday pre-market state
agg_events <- aggregate(evt_df[, event_cols], by = list(timestamp = evt_df$mapped_trading_day), FUN = sum, na.rm = TRUE)

# Carry forward continuous sentiment, uncertainty, and GPR levels via zoo::na.locf
cont_cols <- c("gpr_score", "gpr_acts", "gpr_threats", "gpr_log_change", "gpr_zscore_60d",
               "finbert_sentiment", "finbert_uncertainty", "generic_sentiment")
evt_xts <- xts(evt_df[, cont_cols], order.by = evt_df$timestamp)
grid_xts <- merge(evt_xts, xts(, order.by = trading_days), join = "outer")
grid_filled <- na.locf(grid_xts, na.rm = FALSE)
grid_filled <- na.locf(grid_filled, fromLast = TRUE, na.rm = FALSE)
aligned_cont <- data.frame(timestamp = index(grid_filled), coredata(grid_filled))
aligned_cont <- aligned_cont[aligned_cont$timestamp %in% trading_days, ]

aligned_events <- merge(agg_events, aligned_cont, by = "timestamp", all.x = TRUE)
write_parquet(aligned_events, "data/02_interim/r_aligned_events.parquet")
cat(sprintf("[R Refinery 3/3] Saved look-ahead-free aligned events (%d trading days) to data/02_interim/r_aligned_events.parquet\n", nrow(aligned_events)))