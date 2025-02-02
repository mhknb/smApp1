import os
import sys
import pandas as pd     #  version: 1.5.3
import streamlit as st  #  version: 1.41.1
from sklearn.model_selection import train_test_split
from datetime import datetime
import subprocess
from PIL import Image    #  version: 11.0.0
sys.path.append("../") 
from synthesizers.table_eva import TableEvaluatorClass
from table_evaluator import TableEvaluator

st.markdown("""
        <style>
            .stButton>button {
                background-color: #3300FFFF;
                color: white;
                font-size: 18px;
                padding: 15px 32px;
                border: none;
                border-radius: 12px;
                cursor: pointer;
                transition: background-color 0.3s, transform 0.2s;
            }
            .stButton>button:hover {
                background-color: #002AFFFF;
                transform: scale(1.1);
                color: white;
            }
            img {
    width: 100%;        /* Set the width of the image */
    height: auto;       /* Maintain the aspect ratio */
    border-radius: 10px; /* Round corners of the image */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Add a shadow around the image */
    object-fit: cover;   /* Ensures the image fills the container without distortion */
    margin: 20px;        /* Add space around the image */
    transition: transform 0.3s ease; /* Smooth transition on hover */
    }

    img:hover {
    transform: scale(1.05); /* Slight zoom effect on hover */
    }
                    .image-container {
                display: inline-block;
                margin: 10px;
                padding: 10px;
                border-radius: 10px;
                border: 2px solid #ddd;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .image-container:hover {
                transform: scale(1.05);
                box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2);
            }
            .image-container img {
                border-radius: 8px;
                max-width: 100%;
                height: auto;
            }
        </style>
    """, unsafe_allow_html=True)
output_path = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
selected_columns_path = os.path.join(os.getcwd(), "selected_columns.py")
tableEvaluator_path = os.path.join(os.getcwd(), "tableEvaluator")




nav_option = st.sidebar.radio("", ["GAN Model", "Table Evaluator"])




if nav_option == "GAN Model":

    # Veri işleme ve GAN modeli eğitimi
    st.title("Veri İşleme ve GAN Modeli Eğitim Pipeline'ı")

    # Kullanıcılar veri kümesinin yolunu (`data_path`) bir metin kutusuyla girer.
    data_path = st.text_input("Veri seti yolunu giriniz:")
    
    cat_cols=[]
    cont_cols=[]
    ord_cols=[]
    cat_method =""
    cont_method=""
    target_col=""
    
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path ,low_memory=False)
        st.write("Orijinal Veri Önizleme:", df.head())
        
        
        # Veri işleme parametreleri seçimi
    
        st.subheader("Veri İşleme Parametrelerini Seçin")
        cat_method = st.selectbox("Kategorik Kodlama Yöntemi:", ["onehot", "label"], index=1, help="Kategorik verileri sayısal değerlere dönüştürmek için kullanılan yöntemleri seçin. 'One-Hot Encoding' her kategori için ayrı bir sütun yaratırken, 'Label Encoding' her kategoriye bir sayı atar.")
        cont_method = st.selectbox("Sürekli Veri Ölçeklendirme Yöntemi:", ["bayesian", "standard", "minmax", "none"], index=1, help="Sürekli sayısal verilerin ölçeklendirilmesi için yöntemler. 'Standard Scaling' verileri ortalama 0 ve standart sapma 1 olacak şekilde ölçeklendirirken, 'Min-Max Scaling' verileri 0 ile 1 arasında ölçeklendirir.")
        ord_method = st.selectbox("Sıralı Veri Kodlama Yöntemi:", ["Ordinal Encoding"], index=0, help="Sıralı kategorik verileri sayısal değerlere dönüştürmek için 'Ordinal Encoding' yöntemini kullanabilirsiniz.")

        
        # Her tür veri için (kategorik, sürekli, sıralı) uygun sütunları seçer ve hedef sütunu (bağımlı değişken) belirtir.
        cat_cols = st.multiselect("Seçin Kategorik Sütunlar", df.columns, help="Farklı kategorileri veya sınıfları tzemsil eden, sayısal olmayan değerlerdir, örneğin cinsiyet (Erkek/Kadın) veya ülke (Türkiye, Almanya) gibi. Bu değerler sayısal değildir.")
        cont_cols = st.multiselect("Sürekli Sütunları Seçin:", df.columns , help="Sayısal değerler, sürekli sütunları (örneğin boy veya gelir) ve kesikli sütunları (örneğin sayım veya yaş) içerir.")
        ord_cols = st.multiselect("Sıralı Sütunları Seçin:", df.columns , help="Ayrık veya sayısal değerler. Bazı durumlarda, bu değerler sayısal olabilir, örneğin sayım (bir ailedeki çocuk sayısı veya ürün sayısı gibi), ve bu sütunlar genellikle sayısal olup sayılabilen değerlere sahiptir.")

        target_col = st.selectbox("Target Column: Daha önce seçili sütunlardan Hedef sütununu seçin.", (cat_cols + cont_cols + ord_cols))
        if target_col not in (cat_cols + cont_cols + ord_cols):
            st.error("Hedef sütunu, daha önce seçilen kategorik, sürekli veya sıralı sütunlardan birisi olmalıdır!")


        if st.button("Seçimleri Kaydet"):
            selected_data = {
                'cat_cols': cat_cols,
                'cont_cols': cont_cols,
                'ord_cols': ord_cols,
                'target_col': [target_col],
                
            }
            
            # Seçimleri bir Python dosyasına yaziyor(selected_columns.py)
            with open(selected_columns_path, 'w') as f:
                f.write("cat_cols = " + str(selected_data['cat_cols']) + "\n")
                f.write("cont_cols = " + str(selected_data['cont_cols']) + "\n")
                f.write("ord_cols = " + str(selected_data['ord_cols']) + "\n")
                f.write("target_col = " + str(selected_data['target_col']) + "\n")

            st.success(f" Seçimler kaydedildi:   {selected_columns_path}")
        



        # Apply the pipeline
        try:

            transformed_df = pd.DataFrame(df[cat_cols + cont_cols + ord_cols])

            st.write("Yeni veri seti:", transformed_df.head())

            """
            Eğitim-Test Bölmesi:
                - Kullanıcıların test için kullanılacak veri oranını belirtmeleri için bir kaydırıcı sağlanır.
                - Veri kümesi, (train_test_split) kullanılarak eğitim ve test alt kümesine bölünür.
                - Eğitim ve test veri kümelerinin önizlemeleri gösterilir.
            """


            # Train-Test Split
            test_size = st.slider("Test Size (fraction):", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
            random_state = st.number_input("Random State:", value=123, step=1)

            train_data, test_data = train_test_split(transformed_df, test_size=test_size, random_state=random_state)
            st.write("Eğitim Verisi Önizlemesi:", train_data.head())
            st.write("Test Verisi Önizlemesi:", test_data.head())

            # base_path = st.text_input("Enter the base path for saving the data (without extension):",  os.path.join(os.path.dirname(os.getcwd()) ,'data',"new\\" ) )
            base_path =    os.path.join(os.path.dirname(os.getcwd()) ,'data',"new/" ) 

            if st.button("Split Data Kaydet"):
    
                if not os.path.exists(base_path):
                    os.makedirs(base_path)
                train_path = f"{base_path}train.csv"  
                test_path = f"{base_path}test.csv"   
                transformed_data_path = f"{base_path}new.csv"  # Tam dönüştürülmüş veri kümesi  new.csv

                # Bölünmüş Veriyi Kaydetme
                train_data.to_csv(train_path, index=False)
                test_data.to_csv(test_path, index=False)
                
                # Dönüştürülmüş verisetini kaydet
                transformed_df.to_csv(transformed_data_path, index=False)
                
                st.success(f"Training data saved at {train_path}, testing data saved at {test_path}, and transformed data saved at {transformed_data_path}!")



        except Exception as e:
            st.error(f"Error during data transformation or splitting: {e}")
    else:
        st.warning("Please provide a valid dataset path.")

    # GAN Training Configuration
    st.subheader("GAN Model Training")
    model_name = st.selectbox('GAN Modelini Seçin :', ['tablegan', 'sdv_tvae'], index=0)
    epochs = st.number_input('Epok:', min_value=1, max_value=100, value=2)
    seed = st.number_input('Rastgele Seed:', min_value=1, max_value=1000, value=123)
    subset_size = st.number_input('Alt Küme Boyutu:', min_value=1, max_value=100000, value=1000)
    sample_size = st.number_input('Sample Boyutu:', min_value=1, max_value=10000, value=1000)
    eval_retries = st.number_input('Değerlendirme Yeniden Deneme Sayısı:', min_value=1, max_value=10, value=3)


    tabpath = os.getcwd()
    tablegan_path = os.path.join(os.getcwd(), 'tablegan')
    tvae_path = os.path.join(os.getcwd(), 'tvae')



    if model_name == 'tablegan':
        model_path = 'tablegan.py'
        working_directory = tablegan_path
    elif model_name == 'sdv_tvae':
        model_path = 'sdv_tvae.py'
        working_directory = tvae_path
    else:
        raise ValueError(f"Invalid model name: {model_name}")



    saved_name = f"{model_name}_{output_path}"
    output_directory = f"{working_directory}/results/{saved_name}"


    #  GAN Komutunu Tanımla 
    command = [
        'python', model_path  ,
        '-name', saved_name,
        '-data', "new",
        '-ep', str(epochs),
        '-s', str(seed),
        '--train', "True",
        '--evaluate', "True",
        '--sample', "True",
        '--subset_size', str(subset_size),
        '--sample_size', str(sample_size),
        '--eval_retries', str(eval_retries),
        '--discrete_preprocess', cat_method,
        '--numerical_preprocess', cont_method,
        '-target',target_col,
    
    ]
    # Komutu çalıştırma
    if st.button('Komutu Çalıştır'):
        try:
            # Komutu subprocess ile çalıştır
            result = subprocess.run(command, check=True, text=True, capture_output=True, cwd=working_directory)

            # result = subprocess.run(command, check=True, text=True, capture_output=True)
            st.success('Komut başarıyla çalıştırıldı!')
        # Çıktıyı göster
            st.text_area("Komut Çıktısı:", result.stdout, height=300)

            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            
            #  Log Dosyasını Göster 
            log_path = os.path.join(output_directory, 'log.txt')
            
            if os.path.exists(log_path):
                with open(log_path, 'r') as log_file:
                    log_content = log_file.readlines()


                data_rows = []
                for line in log_content:
                    if line.strip():  # Skip empty lines
                        # Split each line by spaces/tabs and store the data
                        data_rows.append(line.split())

                # Create a DataFrame
                df = pd.DataFrame(data_rows)

                # Veri Çerçevesini Streamlit'te tablo olarak göster
                st.subheader('Eğitim Günlüğü Tablo Olarak:')
                st.table(df)


            
            # Resimleri Göster
            image_files = [f for f in os.listdir(output_directory) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            for image_file in image_files:
                image_path = os.path.join(output_directory, image_file)
                image = Image.open(image_path)
                st.image(image, caption=image_file, use_container_width=True)
    
            
            
        except subprocess.CalledProcessError as e:
            st.error(f"Komut çalıştırılırken hata oluştu: {e}")
            print(f"Command failed with error: {e}")
            print(f"Error message: {e.stderr}")









elif nav_option == "Table Evaluator":
    st.title("Table Evaluator")
    
    real_data_df = None
    fake_data_df = None

    # Dosya yolları için giriş alanları
    real_data_path = st.text_input("Gerçek veri CSV dosyasının yolunu girin:")
    if os.path.exists(real_data_path):
        real_data_df = pd.read_csv(real_data_path )
        st.write("Gerçek Veri:", real_data_df.head())
    else:
        st.warning("Lütfen geçerli bir veri kümesi yolu sağlayın.")
        
    fake_data_path = st.text_input("Fake veri CSV dosyasının yolunu girin:")
    if os.path.exists(fake_data_path):
        fake_data_df = pd.read_csv(fake_data_path )
        st.write("Fake Veri:", fake_data_df.head())
    else:
        st.warning("Lütfen geçerli bir veri kümesi yolu sağlayın.")
        
    # Sahte veri yüklendiğinde hedef sütun seçimini göster.
    if fake_data_df is not None:
        target_col = st.selectbox("Target column seçin:", fake_data_df.columns)
    else:
        target_col = None

    if st.button("Tabloları Değerlendir"):
        if real_data_df is not None and fake_data_df is not None and target_col:
            try:
                
                
                evaluator = TableEvaluatorClass(tableEvaluator_path, target_col, real_data_df, fake_data_df)
                evaluator.evaluate()  # Değerlendirmeyi çalıştır
                
                evaluation_results_path = os.path.join(tableEvaluator_path, 'evaluation_results.txt')
            
                if os.path.exists(evaluation_results_path):
                    with open(evaluation_results_path, 'r') as file:
                        results = file.read()
                
                    st.text_area("Sonuçlar:", results, height=300)
   
                
                image_files = [f for f in os.listdir(tableEvaluator_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
                for image_file in image_files:
                    image_path = os.path.join(tableEvaluator_path, image_file)
                    image = Image.open(image_path)
                    st.image(image, caption=image_file, use_container_width=True)
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Lütfen tüm gerekli girişleri sağlayın")
