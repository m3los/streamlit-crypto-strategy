
def convert_tf(tf):
    if "m" in tf:
        return tf[:-1]
    elif "h" in tf:
        return str(int(tf[:-1]) * 60)
    elif "d" in tf:
        return tf[1:]


def convert_sym(exchange, symbols):
    if not symbols:
        return exchange.upper() + "\:BTCUSDT"
    else:
        return exchange.upper() + "\:" + symbols[0].replace("/", "")


def chart_widget(exchange, watchlist_symbols, timeframe):
    widget = f"""<!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container" style="height:100%";width:100%>
      <div id="tradingview_f93a3" style="height:calc(100% - 32px);width:100%"></div>
      <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/" rel="noopener nofollow" target=
      "_blank"><span class="blue-text">Track all markets on TradingView</span></a></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
      "autosize": false,
      "symbol": "{convert_sym(exchange, watchlist_symbols)}",
      "interval": "{convert_tf(timeframe)}",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "toolbar_bg": "#f1f3f6",
      "enable_publishing": false,
      "hide_side_toolbar": false,
      "allow_symbol_change": true,
      "watchlist": {[sym.replace('/', '') for sym in watchlist_symbols]},
      "studies": [
        {{
          id: "RSI@tv-basicstudies",
          inputs: {{
          length: 14
          }}
        }},
        {{
          id: "MASimple@tv-basicstudies",
          inputs: {{
          length: 200
          }}
        }}
      ],
      "container_id": "tradingview_f93a3"
    }}
      );
      </script>
    </div>
    <!-- TradingView Widget END -->"""
    return widget
