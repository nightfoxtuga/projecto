import os
import csv
import random
import argparse
from pathlib import Path

class PollutionDataGenerator:
    def __init__(self):
        # Definir intervalos realistas para cada poluente por categoria de qualidade do ar
        self.pollution_levels = {
            "1_Good": {
                "CO": (0.0, 0.5),        # ppm
                "CO2": (300, 400),       # ppm
                "PM2_5": (0.0, 10.0),    # μg/m³
                "PM10": (0.0, 10),       # μg/m³
                "NO2": (0.0, 53),        # ppb
                "O3": (0.0, 54),         # ppb
                "SO2": (0.0, 35),        # ppb
                "temperature": (15.0, 25.0),  # °C
                "humidity": (40.0, 70.0),     # %
                "pressure": (1010, 1020)      # hPa
            },
            "2_Moderate": {
                "CO": (0.5, 1),
                "CO2": (450, 600),
                "PM2_5": (10, 50),
                "PM10": (10, 50),
                "NO2": (54, 100),
                "O3": (55, 70),
                "SO2": (36, 75),
                "temperature": (20.0, 30.0),
                "humidity": (35.0, 65.0),
                "pressure": (1005, 1015)
            },
            "3_Unhealthy_For_Sensitive_Groups": {
                "CO": (0.5, 1),
                "CO2": (600, 800),
                "PM2_5": (50, 100),
                "PM10": (50, 100),
                "NO2": (101, 360),
                "O3": (71, 85),
                "SO2": (76, 185),
                "temperature": (25.0, 35.0),
                "humidity": (30.0, 60.0),
                "pressure": (1000, 1010)
            },
            "4_Unhealthy": {
                "CO": (1, 3),
                "CO2": (800, 1000),
                "PM2_5": (100.0, 200.0),
                "PM10": (100, 200),
                "NO2": (361, 649),
                "O3": (86, 105),
                "SO2": (186, 304),
                "temperature": (30.0, 40.0),
                "humidity": (25.0, 55.0),
                "pressure": (995, 1005)
            },
            "5_Very_Unhealthy": {
                "CO": (3, 7),
                "CO2": (1000, 1500),
                "PM2_5": (200.0, 400.0),
                "PM10": (200, 400),
                "NO2": (650, 1249),
                "O3": (106, 200),
                "SO2": (305, 604),
                "temperature": (35.0, 45.0),
                "humidity": (20.0, 50.0),
                "pressure": (990, 1000)
            }
        }
        
        # Unidades de medida para cada poluente
        self.units = {
            "CO": "ppm",
            "CO2": "ppm", 
            "PM2_5": "μg/m³",
            "PM10": "μg/m³",
            "NO2": "ppb",
            "O3": "ppb",
            "SO2": "ppb",
            "temperature": "°C",
            "humidity": "%",
            "pressure": "hPa"
        }
    
    def generate_sensor_data(self, pollution_category):
        """Gera dados de sensores realistas para uma categoria de poluição"""
        if pollution_category not in self.pollution_levels:
            raise ValueError(f"Categoria de poluição desconhecida: {pollution_category}")
        
        data = {}
        for pollutant, (min_val, max_val) in self.pollution_levels[pollution_category].items():
            # Gerar valor aleatório dentro do intervalo, com distribuição normal próxima à média
            mean = (min_val + max_val) / 2
            std_dev = (max_val - min_val) / 6  # 99.7% dos valores dentro do intervalo
            
            # Garantir que o valor está dentro dos limites
            value = random.gauss(mean, std_dev)
            value = max(min_val, min(max_val, value))
            
            # Arredondar para uma casa decimal apropriada
            if pollutant in ["CO", "PM2_5", "temperature", "humidity"]:
                value = round(value, 1)
            else:
                value = round(value)
            
            data[pollutant] = value
        
        return data
    
    def process_images_by_folder(self, base_folder, output_file):
        """Processa imagens organizadas por pastas de categorias de poluição"""
        categories = ["1_Good", "2_Moderate", "3_Unhealthy_For_Sensitive_Groups", 
                     "4_Unhealthy", "5_Very_Unhealthy"]
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        csv_data = []
        total_images = 0
        
        for category in categories:
            category_folder = os.path.join(base_folder, category)
            
            if not os.path.exists(category_folder):
                print(f"Aviso: Pasta {category} não encontrada")
                continue
            
            # Encontrar todas as imagens na pasta da categoria
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(category_folder).glob(f"*{ext}"))
                image_files.extend(Path(category_folder).glob(f"*{ext.upper()}"))
            
            print(f"Encontradas {len(image_files)} imagens na categoria {category}")
            
            # Processar cada imagem
            for image_path in image_files:
                # Gerar dados de sensores
                sensor_data = self.generate_sensor_data(category)
                
                # Adicionar metadados
                row = {
                    "image_filename": image_path.name,
                    "pollution_category": category,
                    "image_path": str(image_path)
                }
                
                # Adicionar dados dos sensores
                for pollutant, value in sensor_data.items():
                    row[pollutant] = value
                    row[f"{pollutant}_unit"] = self.units[pollutant]
                
                csv_data.append(row)
                total_images += 1
        
        # Escrever ficheiro CSV
        if csv_data:
            # Obter todos os campos possíveis
            fieldnames = ["image_filename", "pollution_category", "image_path"]
            for pollutant in self.units.keys():
                fieldnames.extend([pollutant, f"{pollutant}_unit"])
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            print(f"\nDados guardados em {output_file}")
            print(f"Total de imagens processadas: {total_images}")
            
            # Mostrar estatísticas por categoria
            print("\nEstatísticas por categoria:")
            for category in categories:
                count = sum(1 for row in csv_data if row["pollution_category"] == category)
                print(f"{category}: {count} imagens")
        else:
            print("Nenhuma imagem foi encontrada ou processada")
        
        return csv_data
    
    def generate_summary_report(self, csv_data, report_file):
        """Gera um relatório sumário com estatísticas dos dados"""
        if not csv_data:
            return
        
        categories = ["1_Good", "2_Moderate", "3_Unhealthy_For_Sensitive_Groups", 
                     "4_Unhealthy", "5_Very_Unhealthy"]
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE DADOS DE POLUIÇÃO GERADOS\n")
            f.write("=" * 50 + "\n\n")
            
            # Estatísticas por categoria
            f.write("IMAGENS POR CATEGORIA:\n")
            for category in categories:
                count = sum(1 for row in csv_data if row["pollution_category"] == category)
                f.write(f"{category}: {count} imagens\n")
            
            f.write("\n" + "=" * 50 + "\n\n")
            
            # Valores médios por categoria e poluente
            f.write("VALORES MÉDIOS POR CATEGORIA:\n")
            pollutants = [p for p in self.units.keys() if p not in ['temperature', 'humidity', 'pressure']]
            
            for category in categories:
                f.write(f"\n{category}:\n")
                category_data = [row for row in csv_data if row["pollution_category"] == category]
                
                if not category_data:
                    continue
                
                for pollutant in pollutants:
                    values = [row[pollutant] for row in category_data]
                    avg = sum(values) / len(values)
                    f.write(f"  {pollutant}: {avg:.2f} {self.units[pollutant]}\n")

def main():
    parser = argparse.ArgumentParser(description='Gerar dados de sensores para imagens de poluição organizadas por pastas')
    parser.add_argument('--input', '-i', required=True, help='Pasta base com as subpastas de categorias de poluição')
    parser.add_argument('--output', '-o', default='pollution_sensor_data.csv', help='Ficheiro CSV de saída')
    parser.add_argument('--report', '-r', default='pollution_data_report.txt', help='Ficheiro de relatório')
    
    args = parser.parse_args()
    
    # Verificar se a pasta base existe
    if not os.path.exists(args.input):
        print(f"Erro: A pasta {args.input} não existe")
        return
    
    # Processar imagens
    generator = PollutionDataGenerator()
    csv_data = generator.process_images_by_folder(args.input, args.output)
    
    # Gerar relatório
    if csv_data:
        generator.generate_summary_report(csv_data, args.report)
        print(f"Relatório gerado em {args.report}")

if __name__ == "__main__":
    main()