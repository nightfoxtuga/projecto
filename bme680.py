import time
import smbus2
from datetime import datetime

class BME680_Simple:
    def __init__(self, bus_num=1, address=0x77):
        self.bus = smbus2.SMBus(bus_num)
        self.address = address
        
        # Registradores do BME680
        self.REG_CHIP_ID = 0xD0
        self.REG_RESET = 0xE0
        self.REG_CTRL_MEAS = 0x74
        self.REG_CTRL_HUM = 0x72
        self.REG_CTRL_GAS_1 = 0x71
        self.REG_GAS_WAIT_0 = 0x64
        self.REG_RES_HEAT_0 = 0x5A
        self.REG_IDAC_HEAT_0 = 0x50
        
        # Registradores de dados
        self.REG_TEMP_MSB = 0x22
        self.REG_TEMP_LSB = 0x23
        self.REG_TEMP_XLSB = 0x24
        self.REG_PRESS_MSB = 0x1F
        self.REG_PRESS_LSB = 0x20
        self.REG_PRESS_XLSB = 0x21
        self.REG_HUM_MSB = 0x25
        self.REG_HUM_LSB = 0x26
        self.REG_GAS_R_MSB = 0x2A
        self.REG_GAS_R_LSB = 0x2B
        
        # Verifica se o sensor está presente
        try:
            chip_id = self.read_byte(self.REG_CHIP_ID)
            if chip_id == 0x61:  # Chip ID do BME680
                print("✅ BME680 detectado no endereço 0x77")
                self.setup_sensor()
            else:
                print(f"❌ Chip ID incorreto: 0x{chip_id:02x}")
                self.bus = None
        except Exception as e:
            print(f"❌ Erro de comunicação: {e}")
            self.bus = None
    
    def read_byte(self, reg):
        """Lê um byte do registrador"""
        return self.bus.read_byte_data(self.address, reg)
    
    def write_byte(self, reg, value):
        """Escreve um byte no registrador"""
        self.bus.write_byte_data(self.address, reg, value)
    
    def setup_sensor(self):
        """Configura o sensor para leitura básica"""
        try:
            # Configuração básica - desativa medição de gás inicialmente
            self.write_byte(self.REG_CTRL_HUM, 0x01)   # Umidade oversampling 1x
            self.write_byte(self.REG_CTRL_MEAS, 0x24)  # Temp 1x, Pressão 1x
            self.write_byte(self.REG_CTRL_GAS_1, 0x00)  # Desativa gás inicialmente
            
            print("✅ Sensor configurado para leitura básica")
            
        except Exception as e:
            print(f"⚠️  Aviso na configuração: {e}")
    
    def read_raw_data(self):
        """Lê dados brutos do sensor"""
        try:
            # Força uma medição
            ctrl_meas = self.read_byte(self.REG_CTRL_MEAS)
            self.write_byte(self.REG_CTRL_MEAS, (ctrl_meas & 0xFC) | 0x01)
            
            # Aguarda conversão
            time.sleep(0.1)
            
            # Lê dados de temperatura (20 bits)
            temp_msb = self.read_byte(self.REG_TEMP_MSB)
            temp_lsb = self.read_byte(self.REG_TEMP_LSB)
            temp_xlsb = self.read_byte(self.REG_TEMP_XLSB)
            temp_raw = (temp_msb << 12) | (temp_lsb << 4) | (temp_xlsb >> 4)
            
            # Lê dados de pressão (20 bits)
            press_msb = self.read_byte(self.REG_PRESS_MSB)
            press_lsb = self.read_byte(self.REG_PRESS_LSB)
            press_xlsb = self.read_byte(self.REG_PRESS_XLSB)
            press_raw = (press_msb << 12) | (press_lsb << 4) | (press_xlsb >> 4)
            
            # Lê dados de umidade (16 bits)
            hum_msb = self.read_byte(self.REG_HUM_MSB)
            hum_lsb = self.read_byte(self.REG_HUM_LSB)
            hum_raw = (hum_msb << 8) | hum_lsb
            
            return {
                'temp_raw': temp_raw,
                'press_raw': press_raw,
                'hum_raw': hum_raw
            }
            
        except Exception as e:
            print(f"❌ Erro na leitura: {e}")
            return None
    
    def compensate_temperature(self, raw_temp):
        """Compensação simplificada de temperatura"""
        # Fórmula aproximada para temperatura
        return round((raw_temp / 100) - 273.15, 2)
    
    def compensate_pressure(self, raw_press):
        """Compensação simplificada de pressão"""
        return round(raw_press / 100, 2)
    
    def compensate_humidity(self, raw_hum):
        """Compensação simplificada de umidade"""
        return round(raw_hum / 1024, 2)
    
    def read_all_data(self):
        """Lê e compensa todos os dados"""
        if not self.bus:
            return None
            
        raw_data = self.read_raw_data()
        if not raw_data:
            return None
        
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'temperature': self.compensate_temperature(raw_data['temp_raw']),
                'pressure': self.compensate_pressure(raw_data['press_raw']),
                'humidity': self.compensate_humidity(raw_data['hum_raw']),
                'temp_raw': raw_data['temp_raw'],
                'press_raw': raw_data['press_raw'],
                'hum_raw': raw_data['hum_raw']
            }
        except Exception as e:
            print(f"❌ Erro na compensação: {e}")
            return None

def print_data(data):
    """Imprime os dados de forma formatada"""
    if not data:
        return
        
    print("\n" + "═" * 50)
    print("🌡️  LEITURA BME680")
    print("═" * 50)
    print(f"🕒 {data['timestamp']}")
    print(f"🌡️  Temperatura: {data['temperature']} °C")
    print(f"📊 Pressão: {data['pressure']} hPa")
    print(f"💧 Umidade: {data['humidity']} %")
    print("═" * 50)
    print(f"📋 Brutos - Temp: {data['temp_raw']}, Press: {data['press_raw']}, Hum: {data['hum_raw']}")

def main():
    print("🚀 Iniciando Leitor BME680")
    print("🔍 Sensor detectado no endereço 0x77")
    
    # Inicializa sensor
    sensor = BME680_Simple(1, 0x77)
    
    if not sensor.bus:
        print("❌ Não foi possível inicializar o sensor")
        return
    
    try:
        print("\n🎯 Iniciando leituras (Ctrl+C para parar)")
        print("═" * 50)
        
        for i in range(10):  # 10 leituras de teste
            data = sensor.read_all_data()
            if data:
                print(f"\n📊 Leitura #{i+1}")
                print_data(data)
            else:
                print(f"⏳ Tentativa {i+1}: Aguardando dados...")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Leitura interrompida pelo usuário")
    except Exception as e:
        print(f"❌ Erro: {e}")
    finally:
        if sensor.bus:
            sensor.bus.close()
        print("✅ Conexão I2C fechada")

# Teste rápido
if __name__ == "__main__":
    main()