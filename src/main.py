import pandas as pd
import visulize as vis

# Load the data from the provided CSV file
file_path = './data/carbonmonitor-cities_datas_2024-09-01.csv'
data = pd.read_csv(file_path)

#vis.draw_emissions_heatmap(data, "New York")
vis.draw_emissions_trends(data, "New York")
