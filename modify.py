import pandas as pd
import numpy as np

# Read the CSV file and clean column names
df = pd.read_csv('beijing-air-quality.csv')
df.columns = df.columns.str.strip()

# Converter a coluna pm25 para numérico
df['pm25'] = pd.to_numeric(df['pm25'], errors='coerce')

# Function to calculate CO2 based on exact PM2.5 to CO2 mapping
def calculate_co2(pm25):
    if pd.isna(pm25):
        return np.nan
    
    # Mapeamento linear entre PM2.5 e CO2 baseado nas faixas
    if pm25 <= 10.0:
        # Good: 300-400 ppm para 0-10 μg/m³
        co2 = 300 + (pm25 / 10.0) * 100
    elif pm25 <= 50.0:
        # Moderate: 450-600 ppm para 10-50 μg/m³
        co2 = 450 + ((pm25 - 10) / 40.0) * 150
    elif pm25 <= 100.0:
        # Unhealthy for Sensitive: 600-800 ppm para 50-100 μg/m³
        co2 = 600 + ((pm25 - 50) / 50.0) * 200
    elif pm25 <= 200.0:
        # Unhealthy: 800-1000 ppm para 100-200 μg/m³
        co2 = 800 + ((pm25 - 100) / 100.0) * 200
    else:
        # Very Unhealthy: 1000-1500 ppm para 200+ μg/m³
        co2 = 1000 + min(((pm25 - 200) / 200.0) * 500, 500)
    
    # Adicionar variação aleatória (±10%)
    variation = np.random.uniform(0.9, 1.1)
    co2 = co2 * variation
    
    # Arredondar para número inteiro (sem casas decimais)
    return round(co2)

# Apply the function
df['co2'] = df['pm25'].apply(calculate_co2)

# Reorder columns
columns = ['date', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'co2']
df = df[columns]

# Save the updated CSV
df.to_csv('beijing-air-quality-with-co2.csv', index=False)

print("CSV file with CO2 values has been created successfully!")
print(f"Added CO2 values for {len(df)} entries")
print(f"CO2 range: {df['co2'].min()} - {df['co2'].max()} ppm")

# Mostrar alguns exemplos para verificar o arredondamento
print("\nPrimeiros 10 valores de CO2 (arredondados):")
print(df['co2'].head(10).tolist())