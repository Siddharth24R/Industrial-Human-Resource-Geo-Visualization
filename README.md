# Industrial Human Resource Geo-Visualization Dashboard

This project visualizes the distribution of human resources across various industrial sectors in India using a Streamlit dashboard. It incorporates geo-location-based worker counts, industry information, clustering of industries, and feature distributions, enabling users to explore workforce data in a structured and interactive way.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
5. [Technologies Used](#technologies-used)

## Project Overview

This dashboard provides insights into the distribution and composition of main and marginal workers in various Indian industrial sectors. It includes the following:
- Worker counts by state and district.
- Industry information and clustering using text analysis.
- Data visualizations for workforce distribution, feature distributions, and correlation analysis.

The data is sourced from `cleaned_combined_data.csv` and includes industry, demographic, and geographic information.

## Features

### 1. **State and District Selection**
   - Users can select a specific **state** and **district** to view relevant workforce data.

### 2. **Worker and Industry Information**
   - Displays total counts of **main** and **marginal workers** by gender, along with industry details.

### 3. **Industry Clustering**
   - Clusters industries using **TF-IDF** and **KMeans**, providing insights into similar industry types based on textual descriptions.

### 4. **Worker Distribution by Industry**
   - Visualizes distribution of main and marginal workers in various industries, with the option to view urban or rural worker data.

### 5. **Feature Distribution**
   - Provides histogram visualizations of various numerical features to understand data distributions.

### 6. **Correlation Matrix**
   - Displays a correlation matrix for numerical features, helping identify potential relationships between different workforce characteristics.

## Setup and Installation

### Prerequisites
- Python 3.7 or above
- Required Python libraries: `pandas`, `numpy`, `streamlit`, `plotly`, `scikit-learn`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/industrial-hr-geo-visualization.git
   cd industrial-hr-geo-visualization
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the data file (`cleaned_combined_data.csv`) is in the specified path:
   ```text
   C:\ALL folder in desktop\PycharmProjects\GUVI-Ai\Resource Management\cleaned_combined_data.csv
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open the provided URL in a web browser.

### Dashboard Sections

- **Select State and District**: Choose specific locations to view relevant workforce data.
- **Worker and Industry Information**: Explore total worker counts and industry data in the selected district.
- **Industry Clustering**: Visualize clusters of similar industries based on TF-IDF vectorization.
- **Worker Distribution by Industry**: Select worker types and industries to view distribution charts.
- **Feature Distribution**: Visualize distributions of numeric features in the dataset.
- **Correlation Matrix**: Explore relationships between numeric features using a correlation heatmap.

## Technologies Used
- **Python Libraries**: 
  - `Streamlit` for building the web app interface.
  - `Pandas` and `NumPy` for data manipulation.
  - `Plotly` for interactive visualizations.
  - `scikit-learn` for clustering and scaling.

---
