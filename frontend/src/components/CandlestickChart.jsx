import { useEffect, useRef } from 'react'
import { createChart, CandlestickSeries } from 'lightweight-charts'

/**
 * CandlestickChart - TradingView Lightweight Charts wrapper.
 * Expects ohlcv: [{ time, open, high, low, close, volume? }]
 * time: ISO string "YYYY-MM-DDTHH:mm:ssZ" or "YYYY-MM-DD"
 * livePrice: optional - updates last bar for live tick (REST fallback)
 * liveKline: optional - { time (ms), open, high, low, close, isClosed } from Binance WebSocket
 */
export default function CandlestickChart({ ohlcv = [], symbol, height = 400, livePrice, liveKline }) {
  const chartRef = useRef(null)
  const chart = useRef(null)
  const candleSeries = useRef(null)

  useEffect(() => {
    if (!chartRef.current) return

    const ch = createChart(chartRef.current, {
      layout: {
        background: { type: 'solid', color: '#0f172a' },
        textColor: '#94a3b8',
      },
      grid: {
        vertLines: { color: '#1e293b' },
        horzLines: { color: '#1e293b' },
      },
      rightPriceScale: {
        borderColor: '#334155',
        scaleMargins: { top: 0.1, bottom: 0.2 },
      },
      timeScale: {
        borderColor: '#334155',
        timeVisible: true,
        secondsVisible: false,
      },
    })

    const series = ch.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
    })

    chart.current = ch
    candleSeries.current = series

    return () => {
      ch.remove()
      chart.current = null
      candleSeries.current = null
    }
  }, [symbol])

  const toChartTime = (t) => {
    if (typeof t === 'number') return t
    if (typeof t === 'string' && t.includes('T')) return Math.floor(new Date(t).getTime() / 1000)
    if (typeof t === 'string') return t.slice(0, 10)
    return t
  }

  useEffect(() => {
    if (!candleSeries.current || !ohlcv?.length) return

    const data = ohlcv.map((bar) => ({
      time: toChartTime(bar.time),
      open: Number(bar.open),
      high: Number(bar.high),
      low: Number(bar.low),
      close: Number(bar.close),
    }))

    candleSeries.current.setData(data)
    chart.current?.timeScale().fitContent()
  }, [ohlcv])

  useEffect(() => {
    if (!candleSeries.current) return
    if (liveKline) {
      const k = liveKline
      const t = Math.floor((k.time || 0) / 1000)
      if (t <= 0) return
      try {
        candleSeries.current.update({
          time: t,
          open: Number(k.open),
          high: Number(k.high),
          low: Number(k.low),
          close: Number(k.close),
        })
      } catch (_) {}
      return
    }
    if (!ohlcv?.length || livePrice == null) return
    const last = ohlcv[ohlcv.length - 1]
    const t = toChartTime(last.time)
    const open = Number(last.open)
    const close = Number(livePrice)
    const high = Math.max(open, Number(last.high), close)
    const low = Math.min(open, Number(last.low), close)
    try {
      candleSeries.current.update({ time: t, open, high, low, close })
    } catch (_) {}
  }, [livePrice, liveKline, ohlcv])

  return (
    <div className="candlestick-chart" style={{ position: 'relative' }}>
      {symbol && <div className="chart-title">{symbol}</div>}
      <div ref={chartRef} style={{ height: `${height}px`, width: '100%' }} />
    </div>
  )
}
