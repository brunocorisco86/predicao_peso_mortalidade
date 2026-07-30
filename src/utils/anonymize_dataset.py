"""
src/utils/anonymize_dataset.py
--------------------------------
Utilitário de anonimização determinística de dados para o projeto C.Vale.
- Converte número de aviário/fazenda em Hexadecimal de 4 caracteres (caixa alta).
- Converte lote em Hexadecimal de 4 caracteres (caixa alta).
- Formata lote_composto como <aviario_hex>-<lote_hex>.
- Anonimiza nomes de produtores, fazendas, extensionistas e usuários com Faker (pt_BR).
"""

import os
import re
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from faker import Faker

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class DatasetAnonymizer:
    def __init__(self, seed: int = 42):
        self.fake = Faker('pt_BR')
        Faker.seed(seed)
        np.random.seed(seed)
        
        # Mappings
        self.fazenda_map = {}
        self.lote_map = {}
        self.produtor_map = {}
        self.nome_fazenda_map = {}
        self.extensionista_map = {}
        self.usuario_map = {}
        
        # Pre-generate 4-hex pool permutation for collision-free mapping
        # 16^4 = 65,536 values
        hex_pool = [f"{i:04X}" for i in range(1, 65536)]
        np.random.shuffle(hex_pool)
        self.fazenda_hex_pool = hex_pool
        self.fazenda_counter = 0

        lote_pool = [f"{i:04X}" for i in range(1, 65536)]
        np.random.shuffle(lote_pool)
        self.lote_hex_pool = lote_pool
        self.lote_counter = 0

    def get_fazenda_hex(self, raw_fazenda) -> str:
        if pd.isna(raw_fazenda) or raw_fazenda is None or str(raw_fazenda).strip() in ('', 'nan', 'None'):
            return "0000"
        
        # Standardize key (remove float .0 if needed)
        val_str = str(raw_fazenda).strip()
        if val_str.endswith('.0'):
            val_str = val_str[:-2]
            
        if val_str not in self.fazenda_map:
            self.fazenda_map[val_str] = self.fazenda_hex_pool[self.fazenda_counter]
            self.fazenda_counter += 1
            
        return self.fazenda_map[val_str]

    def get_lote_hex(self, raw_lote) -> str:
        if pd.isna(raw_lote) or raw_lote is None or str(raw_lote).strip() in ('', 'nan', 'None'):
            return "0000"
            
        val_str = str(raw_lote).strip()
        if val_str.endswith('.0'):
            val_str = val_str[:-2]
            
        if val_str not in self.lote_map:
            self.lote_map[val_str] = self.lote_hex_pool[self.lote_counter]
            self.lote_counter += 1
            
        return self.lote_map[val_str]

    def transform_lote_composto(self, raw_lote_composto) -> str:
        if pd.isna(raw_lote_composto) or raw_lote_composto is None or str(raw_lote_composto).strip() in ('', 'nan', 'None'):
            return "0000-0000"
            
        val_str = str(raw_lote_composto).strip()
        parts = val_str.split('-')
        
        if len(parts) >= 2:
            faz_hex = self.get_fazenda_hex(parts[0])
            lote_hex = self.get_lote_hex(parts[1])
            return f"{faz_hex}-{lote_hex}"
        else:
            faz_hex = self.get_fazenda_hex(val_str)
            return f"{faz_hex}-0000"

    def get_fake_produtor(self, raw_name) -> str:
        if pd.isna(raw_name) or raw_name is None or str(raw_name).strip() in ('', 'nan', 'None'):
            return "Produtor Anonimizado"
        val_str = str(raw_name).strip()
        if val_str not in self.produtor_map:
            self.produtor_map[val_str] = self.fake.name()
        return self.produtor_map[val_str]

    def get_fake_nome_fazenda(self, raw_name) -> str:
        if pd.isna(raw_name) or raw_name is None or str(raw_name).strip() in ('', 'nan', 'None'):
            return "Fazenda Anonimizada"
        val_str = str(raw_name).strip()
        if val_str not in self.nome_fazenda_map:
            self.nome_fazenda_map[val_str] = f"Granja {self.fake.first_name()}"
        return self.nome_fazenda_map[val_str]

    def get_fake_extensionista(self, raw_name) -> str:
        if pd.isna(raw_name) or raw_name is None or str(raw_name).strip() in ('', 'nan', 'None'):
            return "Extensionista Anonimizado"
        val_str = str(raw_name).strip()
        if val_str not in self.extensionista_map:
            self.extensionista_map[val_str] = self.fake.name()
        return self.extensionista_map[val_str]

    def get_fake_usuario(self, raw_user) -> str:
        if pd.isna(raw_user) or raw_user is None or str(raw_user).strip() in ('', 'nan', 'None'):
            return "user_anon"
        val_str = str(raw_user).strip()
        if val_str not in self.usuario_map:
            self.usuario_map[val_str] = f"user_{self.fake.hexify(text='^^^^')}"
        return self.usuario_map[val_str]

def anonymize_database(db_path: str, anonymizer: DatasetAnonymizer):
    print(f"Anonimizando banco SQLite: {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    tables = [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    
    for table in tables:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        print(f"  - Processando tabela '{table}' ({len(df)} linhas)...")
        
        if 'lote_composto' in df.columns:
            df['lote_composto'] = df['lote_composto'].apply(anonymizer.transform_lote_composto)
            df['fazenda'] = df['lote_composto'].apply(lambda x: str(x).split('-')[0])
        elif 'fazenda' in df.columns:
            df['fazenda'] = df['fazenda'].apply(anonymizer.get_fazenda_hex)
            
        if 'produtor' in df.columns:
            df['produtor'] = df['produtor'].apply(anonymizer.get_fake_produtor)
            
        if 'nome_fazenda' in df.columns:
            df['nome_fazenda'] = df['nome_fazenda'].apply(anonymizer.get_fake_nome_fazenda)
            
        if 'extensionista' in df.columns:
            df['extensionista'] = df['extensionista'].apply(anonymizer.get_fake_extensionista)
            
        if 'id_usurio_criao' in df.columns:
            df['id_usurio_criao'] = df['id_usurio_criao'].apply(anonymizer.get_fake_usuario)
            
        if 'id_usurio' in df.columns:
            df['id_usurio'] = df['id_usurio'].apply(anonymizer.get_fake_usuario)
            
        # Re-save table to SQLite
        df.to_sql(table, conn, if_exists='replace', index=False)
        
    conn.close()
    print("✅ Banco SQLite anonimizado com sucesso!")

def anonymize_csv_files(processed_dir: str, anonymizer: DatasetAnonymizer):
    print(f"Anonimizando arquivos CSV em {processed_dir}...")
    processed_path = Path(processed_dir)
    
    for csv_file in processed_path.glob("*.csv"):
        if csv_file.name.startswith(".~lock"):
            continue
        print(f"  - Anonimizando {csv_file.name}...")
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            
            if 'lote_composto' in df.columns:
                df['lote_composto'] = df['lote_composto'].apply(anonymizer.transform_lote_composto)
                df['fazenda'] = df['lote_composto'].apply(lambda x: str(x).split('-')[0])
            elif 'fazenda' in df.columns:
                df['fazenda'] = df['fazenda'].apply(anonymizer.get_fazenda_hex)
                
            if 'produtor' in df.columns:
                df['produtor'] = df['produtor'].apply(anonymizer.get_fake_produtor)
                
            if 'nome_fazenda' in df.columns:
                df['nome_fazenda'] = df['nome_fazenda'].apply(anonymizer.get_fake_nome_fazenda)
                
            if 'extensionista' in df.columns:
                df['extensionista'] = df['extensionista'].apply(anonymizer.get_fake_extensionista)
                
            if 'id_usurio_criao' in df.columns:
                df['id_usurio_criao'] = df['id_usurio_criao'].apply(anonymizer.get_fake_usuario)
                
            if 'id_usurio' in df.columns:
                df['id_usurio'] = df['id_usurio'].apply(anonymizer.get_fake_usuario)
                
            df.to_csv(csv_file, index=False)
        except Exception as e:
            print(f"    ⚠️ Erro ao processar {csv_file.name}: {e}")
            
    print("✅ Arquivos CSV anonimizados com sucesso!")

def anonymize_raw_excel_files(raw_dir: str, anonymizer: DatasetAnonymizer):
    print(f"Anonimizando arquivos Excel brutos em {raw_dir}...")
    raw_path = Path(raw_dir)
    
    for excel_file in raw_path.rglob("*.xlsx"):
        if excel_file.name.startswith("~") or excel_file.name.startswith(".~lock"):
            continue
        print(f"  - Anonimizando arquivo bruto: {excel_file.relative_to(raw_path)}...")
        try:
            xl = pd.ExcelFile(excel_file)
            sheets_dict = {}
            for sheet in xl.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet)
                
                # Check for lote / lote_composto variations
                for col in ['lote_composto', 'LoteComposto', 'Lote Composto', 'lote composto']:
                    if col in df.columns:
                        df[col] = df[col].apply(anonymizer.transform_lote_composto)
                        
                # Check for fazenda variations
                for col in ['fazenda', 'Fazenda', 'aviario']:
                    if col in df.columns:
                        df[col] = df[col].apply(anonymizer.get_fazenda_hex)
                        
                # Check for produtor variations
                for col in ['produtor', 'Produtor']:
                    if col in df.columns:
                        df[col] = df[col].apply(anonymizer.get_fake_produtor)
                        
                # Check for nome_fazenda variations
                for col in ['nome_fazenda', 'Nome Fazenda']:
                    if col in df.columns:
                        df[col] = df[col].apply(anonymizer.get_fake_nome_fazenda)
                        
                # Check for extensionista variations
                for col in ['extensionista', 'Extensionista']:
                    if col in df.columns:
                        df[col] = df[col].apply(anonymizer.get_fake_extensionista)
                        
                # Check for usuario variations
                for col in ['id_usurio_criao', 'ID Usuário Criação']:
                    if col in df.columns:
                        df[col] = df[col].apply(anonymizer.get_fake_usuario)
                        
                for col in ['id_usurio', 'ID Usuário']:
                    if col in df.columns:
                        df[col] = df[col].apply(anonymizer.get_fake_usuario)
                        
                sheets_dict[sheet] = df
                
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                for sheet, df in sheets_dict.items():
                    df.to_excel(writer, sheet_name=sheet, index=False)
                    
        except Exception as e:
            print(f"    ⚠️ Erro ao processar {excel_file.name}: {e}")
            
    print("✅ Arquivos Excel brutos anonimizados com sucesso!")

def run_all_anonymization():
    anonymizer = DatasetAnonymizer(seed=42)
    
    db_path = str(PROJECT_ROOT / "database" / "prediction_data.db")
    processed_dir = str(PROJECT_ROOT / "data" / "processed")
    raw_dir = str(PROJECT_ROOT / "data" / "raw")
    
    if os.path.exists(raw_dir):
        anonymize_raw_excel_files(raw_dir, anonymizer)

    if os.path.exists(db_path):
        anonymize_database(db_path, anonymizer)
        
    if os.path.exists(processed_dir):
        anonymize_csv_files(processed_dir, anonymizer)

if __name__ == "__main__":
    run_all_anonymization()
