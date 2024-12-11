import streamlit as st

# Sample HTML template with Lightweight-Charts integration
def render_chart(data):
    """
    Renders a candlestick chart using Lightweight-Charts.

    Parameters
    ----------
    data : list
        List of candlestick data in the format:
        [
            {"time": "2024-12-01", "open": 100, "high": 105, "low": 95, "close": 102},
            ...
        ]
    """
    chart_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
    </head>
    <body>
        <div id="chart" style="width: 100%; height: 500px;"></div>
        <script>
            const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
                width: 700,
                height: 500,
                layout: {{
                    backgroundColor: '#ffffff',
                    textColor: '#000000',
                }},
                grid: {{
                    vertLines: {{
                        color: '#e1e1e1',
                    }},
                    horzLines: {{
                        color: '#e1e1e1',
                    }},
                }},
                timeScale: {{
                    timeVisible: true,
                    secondsVisible: false,
                }},
            }});

            const candlestickSeries = chart.addCandlestickSeries();
            candlestickSeries.setData({data});
        </script>
    </body>
    </html>
    """
    return chart_template

# Prepare sample candlestick data
sample_data = [
    {"time": "2024-12-01", "open": 100, "high": 105, "low": 95, "close": 102},
    {"time": "2024-12-02", "open": 102, "high": 110, "low": 100, "close": 108},
    {"time": "2024-12-03", "open": 108, "high": 112, "low": 104, "close": 110},
]


# Render the chart in Streamlit
st.title("Candlestick Chart with Lightweight-Charts")
st.components.v1.html(render_chart(sample_data), height=600)
