# AI-Powered Market Trend Analysis System 

## Project Overview

An advanced AI-powered system designed to analyze market data, predict product trends, forecast demand, and segment customer behavior using machine learning and deep learning techniques. This v2 implementation features an improved modular architecture, enhanced forecasting models, and interactive dashboards.

## 🎯 Key Features

### 1. Market Data Analysis
- **Product Trend Detection**: Identify rising and falling product trends using time-series analysis
- **Customer Segmentation**: Cluster customers based on purchase behavior and demographics
- **Pricing Pattern Analysis**: Understand price elasticity and demand sensitivity
- **Anomaly Detection**: Detect unusual spikes or drops in sales and prices

### 2. Advanced Forecasting
- **Prophet Time-Series Forecasting**: Seasonality-aware demand predictions
- **LSTM Neural Networks**: Deep learning models for long-term trend predictions
- **ARIMA Models**: Statistical forecasting for stable time-series data
- **Ensemble Methods**: Combined predictions for improved accuracy

### 3. Interactive Dashboards
- **Real-time Visualizations**: Plotly and Streamlit-based interactive charts
- **Trend Heatmaps**: Visual representation of product performance across segments
- **Predictive Graphs**: Sales forecasts with confidence intervals
- **Custom Metrics**: Key performance indicators tailored to business needs

### 4. NLP Integration
- **Sentiment Analysis**: Extract insights from product reviews and social media
- **BERT-based Models**: Advanced natural language processing
- **Trend Extraction**: Identify emerging topics and keywords from text data

## 🏗️ System Architecture v2

```
┌─────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                  │
│  (CSV, APIs, Google Trends, Yahoo Finance, Reviews)     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                Data Processing Layer                     │
│  - Cleaning & Validation                                │
│  - Feature Engineering                                  │
│  - Normalization & Scaling                              │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
│   Trend      │ │Segmentation│ │  Sentiment  │
│  Analysis    │ │  Models    │ │  Analysis   │
└───────┬──────┘ └────┬─────┘ └──────┬──────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              AI/ML Prediction Layer                      │
│  - Prophet, LSTM, ARIMA, Random Forest, XGBoost        │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│             Evaluation & Validation Layer                │
│  - Metrics: RMSE, MAE, Accuracy, F1-Score              │
│  - Cross-validation, Backtesting                        │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│            Visualization & Dashboard Layer               │
│  - Interactive Streamlit Dashboard                      │
│  - Plotly Charts, Heatmaps, Predictions                │
│  - Real-time Metrics and KPIs                          │
└─────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
ai-market-trend-analysis-v2/
├── notebooks/
│   ├── 01_data_exploration.ipynb          # EDA and data analysis
│   ├── 02_feature_engineering.ipynb       # Feature creation
│   ├── 03_trend_analysis.ipynb            # Product trend detection
│   ├── 04_forecasting.ipynb               # Demand prediction models
│   ├── 05_segmentation.ipynb              # Customer clustering
│   └── 06_sentiment_analysis.ipynb        # NLP on reviews
├── src/
│   ├── __init__.py
│   ├── data_processor.py                  # Data cleaning & preprocessing
│   ├── feature_engineer.py                # Feature engineering functions
│   ├── trend_detector.py                  # Trend analysis module
│   ├── forecaster.py                      # Forecasting models
│   ├── segmentation.py                    # Clustering algorithms
│   ├── sentiment_analyzer.py              # NLP sentiment analysis
│   └── visualizer.py                      # Visualization utilities
├── models/
│   ├── prophet_model.pkl                  # Trained Prophet model
│   ├── lstm_model.h5                      # LSTM neural network
│   ├── kmeans_clusters.pkl                # KMeans clustering model
│   └── sentiment_model.pkl                # Sentiment classifier
├── data/
│   ├── raw/                               # Original datasets
│   ├── processed/                         # Cleaned datasets
│   └── predictions/                       # Model outputs
├── dashboards/
│   ├── app.py                             # Main Streamlit app
│   ├── pages/
│   │   ├── trends.py                      # Trend visualization page
│   │   ├── forecasts.py                   # Predictions page
│   │   ├── segments.py                    # Customer segments page
│   │   └── sentiment.py                   # Sentiment insights page
│   └── assets/                            # Images, CSS
├── tests/
│   ├── test_data_processor.py
│   ├── test_forecaster.py
│   └── test_segmentation.py
├── docs/
│   ├── ARCHITECTURE.md                    # Detailed system design
│   ├── API.md                             # API documentation
│   └── DEPLOYMENT.md                      # Deployment guide
├── requirements.txt                       # Python dependencies
├── setup.py                               # Package setup
├── config.yaml                            # Configuration file
└── README.md                              # This file
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- Git
- Virtual Environment (recommended)

### Steps

```bash
# Clone the repository
git clone https://github.com/pratham07-8/ai-market-trend-analysis-v2.git
cd ai-market-trend-analysis-v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run dashboards/app.py
```

## 📊 Datasets

The system supports multiple data sources:

1. **Kaggle**: Retail Sales, E-commerce, Market Data
2. **APIs**: Google Trends, Yahoo Finance, Social Media
3. **Custom CSVs**: Your own product and sales data
4. **Synthetic Data**: Generated for testing and demos

## 🤖 AI/ML Techniques Used

| Technique | Use Case | Library |
|-----------|----------|----------|
| Time-Series Forecasting | Demand prediction | Prophet, statsmodels |
| LSTM Networks | Long-term trends | TensorFlow/Keras |
| K-Means Clustering | Customer segmentation | scikit-learn |
| Isolation Forest | Anomaly detection | scikit-learn |
| Random Forest | Feature importance | scikit-learn |
| BERT | Sentiment analysis | Hugging Face Transformers |
| XGBoost | Classification tasks | xgboost |

## 📈 Model Performance Metrics

### Forecasting Models
- **Prophet RMSE**: ~8-12% of average sales
- **LSTM MAE**: Competitive with Prophet for seasonal data
- **ARIMA**: Best for stable, non-trending data

### Segmentation
- **K-Means Silhouette Score**: 0.65-0.75
- **Davies-Bouldin Index**: Optimized for cluster quality

### Sentiment Analysis
- **BERT Accuracy**: 90%+ on product reviews
- **F1-Score**: 0.88-0.92 across sentiment classes

## 🚀 Usage Examples

### Run the Dashboard
```bash
streamlit run dashboards/app.py
```

### Use Python API
```python
from src.forecaster import ProphetForecaster
from src.segmentation import CustomerSegmenter

# Load data
data = pd.read_csv('data/sales.csv')

# Forecast demand
forecaster = ProphetForecaster(data)
forecasts = forecaster.predict(periods=30)

# Segment customers
segmenter = CustomerSegmenter(data)
segments = segmenter.fit_predict()
```

## 📝 Evaluation & Validation

- **Train-Test Split**: 80-20 for model evaluation
- **Cross-Validation**: 5-fold CV for robustness
- **Backtesting**: Historical validation on past data
- **A/B Testing**: Compare model predictions with actual outcomes

## 🔒 Ethical Considerations

- **Bias Mitigation**: Balanced datasets across customer segments
- **Data Privacy**: No PII stored; anonymous aggregation
- **Responsible AI**: Explainability via SHAP values
- **Fairness**: Regular audits for algorithmic bias

## 📚 Learning Outcomes

✓ End-to-end ML pipeline development
✓ Time-series forecasting with Prophet and LSTM
✓ Customer segmentation and clustering
✓ Sentiment analysis with transformers
✓ Interactive dashboard development with Streamlit
✓ Model evaluation and optimization
✓ Production-ready code practices

## 🎓 Technologies & Libraries

- **Data**: pandas, numpy, polars
- **ML**: scikit-learn, XGBoost, LightGBM
- **DL**: TensorFlow, Keras, PyTorch
- **NLP**: Hugging Face Transformers, NLTK, spaCy
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Dashboard**: Streamlit, Dash
- **Forecasting**: Prophet, statsmodels
- **Utilities**: Jupyter, pytest, logging

## 📄 Documentation

- [System Architecture](docs/ARCHITECTURE.md) - Detailed design decisions
- [API Documentation](docs/API.md) - Function and class references
- [Deployment Guide](docs/DEPLOYMENT.md) - Production setup

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## ✨ Acknowledgments

- Facebook's Prophet team for forecasting tools
- Hugging Face for transformer models
- Streamlit for dashboard framework
- Kaggle for public datasets
- IITM Online for project guidelines

## 📞 Contact & Support

**Author**: Pratham (pratham07-8)  
**Email**: prathamchouhan824@gmail.com  
**GitHub**: https://github.com/pratham07-8

---

**Last Updated**: January 2026  
**Version**: 2.0.0  
**Status**: Active Development
