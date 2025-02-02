
import os
import pandas as pd
import streamlit as st
from table_evaluator import TableEvaluator

class TableEvaluatorClass:
    def __init__(self, tableEvaluator_path, target_col, real_data, fake_data):
        self.tableEvaluator_path = tableEvaluator_path
        self.target_col = target_col
        self.real_data = real_data
        self.fake_data = fake_data

    def evaluate(self):
        if not os.path.exists(self.tableEvaluator_path):
            os.makedirs(self.tableEvaluator_path)

        # Değerlendiriciyi başlat
        evaluator = TableEvaluator(self.real_data, self.fake_data, verbose=True)

        # Değerlendirme yap
        evaluator.visual_evaluation(save_dir=self.tableEvaluator_path)
        evaluation_results = evaluator.evaluate(target_col=self.target_col, verbose=True, return_outputs=True)

        # Sonuçları bir metin dosyasına kaydet
        with open(f"{self.tableEvaluator_path}/evaluation_results.txt", "w", encoding="utf-8") as f:
            for section, content in evaluation_results.items():
                f.write(f"{section}:\n")
                for key, value in content.items():
                    if isinstance(value, dict):
                        f.write(f"  {key}:\n")
                        for sub_key, sub_value in value.items():
                            f.write(f"    {sub_key}: {sub_value}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                f.write("\n")

        st.success(f"Değerlendirme tamamlandı, Sonuçlar şu konuma kaydedildi: {self.tableEvaluator_path}/evaluation_results.txt")
