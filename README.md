# Industrial HR Geo-Visualization Dashboard

## 📊 Overview
The **Industrial HR Geo-Visualization Dashboard** is a Streamlit-based interactive web application designed to visualize and analyze human resource distribution across various industries in India. This project integrates geospatial data, visualization tools, and machine learning techniques to provide insights into workforce demographics and their geographic distributions.

## 🌟 Features
- **Choropleth Maps**: Visualize the distribution of workers across Indian states for selected metrics.
- **Gender Distribution**: Explore the male-to-female worker ratio using pie charts and bar graphs.
- **Data Distribution**: Analyze metrics with histograms for better understanding.
- **Clustering**: Perform clustering analysis using TF-IDF and K-Means for industrial categorization.
- **Responsive Design**: User-friendly dashboard with customized visualizations and clean UI.
- **Modular Components**: Easily extendable and maintainable code structure.

## 🚀 Technologies Used
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python (pandas, numpy, scikit-learn, nltk)
- **Visualization**: Plotly
- **Geospatial Data**: GeoJSON for mapping
- **Data Source**: CSV dataset containing industrial and demographic statistics

## 📂 File Structure
```bash
├── app.py                  # Main Streamlit application file
├── cleaned_combined_data.csv # Input dataset for visualization
├── README.md               # Project documentation
   ```

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/industrial-hr-geo-visualization.git
   cd industrial-hr-geo-visualization
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Open your browser at `http://localhost:8501`.

## 📚 How to Use
1. Select a feature to visualize from the sidebar.
2. Interact with choropleth maps, pie charts, and bar graphs to explore insights.
3. Use clustering analysis to categorize industries and view grouped results.

## 📊 Visualizations
1. **Choropleth Maps**:
   Displays the distribution of selected metrics across Indian states.
2. **Gender Distribution**:
   Shows the proportion of male and female workers in various categories.
3. **Clustering Analysis**:
   Groups industries based on similarity for better insights.

## 🤖 Machine Learning Techniques
- **TF-IDF Vectorization**: Converts textual data into meaningful features.
- **K-Means Clustering**: Groups industries into clusters for better understanding.
- **Silhouette Score**: Evaluates clustering quality.

## 🛠️ Development Notes
- Ensure the dataset is present in the specified file path: `cleaned_combined_data.csv`.
- Use the [GeoJSON URL](https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson) for map visualizations.

## 📑 Dependencies
- Streamlit
- pandas
- numpy
- plotly
- scikit-learn
- nltk

Install all required libraries using:
```bash
pip install -r requirements.txt
```

## 🎯 Future Enhancements
- Add more datasets to expand visualization options.
- Enable real-time data updates via API integration.
- Enhance clustering with advanced machine learning models.

---
      
🌟 **Contributors**:Siddharth.R  
📧 Contact: siddharth.r.college@gmail.com 
