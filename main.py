import streamlit as st
import tensorflow as tf
import numpy as np

# Tensorflow Model Prediction with Softmax Confidence Thresholding
def model_prediction(test_image, threshold=0.7):
    model = tf.keras.models.load_model('trained_HarvestGuard_Crop_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    
    prediction = model.predict(input_arr)
    
    # highest probability and corresponding index
    max_prob = np.max(prediction)
    result_index = np.argmax(prediction)
    
    # max probability is less than the threshold, and mark it as "Unknown"
    if max_prob < threshold:
        return "Unknown", max_prob
    else:
        return result_index, max_prob

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if(app_mode == "Home"):
    st.header("HARVEST GUARD CROP SYSTEM")
    image_path = "asset/home.jpg"
    st.image(image_path)
    st.markdown("""
    Welcome to the HarvestGuard Crop System! 🌿🔍
    
    Our mission is to help in identifying HarvestGuard Crop System efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **HarvestGuard Crop Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **HarvestGuard Recognition** page in the sidebar to upload an image and experience the power of our HarvestGuard Crop Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Page
elif(app_mode == "About"):
    st.header("About")
    st.markdown(""" 
    ### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on 
    ### Content
    1. Train (27129 images)    
    2. Valid (6451 images)    
    3. Test (90 images) 
    """)

# Prediction Page

elif(app_mode == "Disease Recognition"):
    try:        
        st.header("HARVEST GUARD CROP SYSTEM")
        test_image = st.file_uploader("Choose an Image")        
    
        if(st.button("Predict")):            
            with st.spinner("Please Wait..."):            
                col1, col2 = st.columns(2)
            # Add content to the first column
            with col1:
                if test_image is None:
                    print("please enter Images")
                else:
                    st.header("Prediction message")
                    st.image(test_image)                
                    st.header("Our prediction")
                    result_index, confidence = model_prediction(test_image, threshold=0.7)

                # Define class names
                class_name = ['Potato___Early_blight', 
                              'Potato___Late_blight', 
                              'Potato___healthy', 
                              'Rice___Bacterialblight', 
                              'Rice___Blast', 
                              'Rice___Brownspot', 
                              'Rice___Healthy', 
                              'Rice___Tungro', 
                              'Tomato___Bacterial_spot', 
                              'Tomato___Early_blight', 
                              'Tomato___Late_blight', 
                              'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
                              'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
                              'Tomato___Tomato_mosaic_virus', 
                              'Tomato___healthy']

                
                if result_index == "Unknown":
                    st.error("Unknown crop.")                        
                else:
                    st.success("Disease Name:  {}".format([item.replace("_", " ") for item in class_name][result_index]))
                    st.write(f"Confidence Score: {confidence:.2f}")
                
            # Add content to the second column
        
            with col2:
                try:
                    st.header([item.replace("_", " ") for item in class_name][result_index])
                    if(class_name  [result_index] == "Potato___Early_blight"):
                        st.write(""" 
                                **Potato Early blight** is caused by the fungus **Alternaria solani**. Small, brown, angular spots appear on the lower leaves. The affected area is covered with a small brown area and is followed by relatively long, black circular spots. Yellowing of the plant, leaf fall and premature death of the plant are the symptoms of this disease. Dark brown to black spots appear on the affected tubers.
                             
                                **Early Blight Disease Remedy**                             
                                1. Ensure balanced fertilizer application and timely irrigation.                             
                                2. Apply 2 grams of Roveral per liter of water as soon as the disease appears and after 7-10 days. Dithane M-45 can be applied at a rate of 0.2%.                             
                                3. Cultivate early varieties of potatoes.
                             """)
                    elif(class_name  [result_index] == "Potato___Late_blight"):
                        st.write(""" 
                                **Potato Late Blight** is a serious disease that affects the leaves, stems, and fruits of potato plants. It is caused by a fungal disease called Phytophthora infestans. The disease can cause significant damage to potato production and, if not controlled, can lead to complete crop failure.
                             
                                **Things to do before the disease occurs**                             
                                1. Regular field inspections and weather monitoring should be carried out during the potato season.                             
                                2. As soon as the weather forecast is favorable for the disease, spray the plants thoroughly with an approved fungicide of the Mancozeb group such as Dithane M-45 or Indofil M-45 or Pencozeb 80 WP at the rate of 2 grams per liter of water every 7 days to prevent the disease. Otherwise, the disease can become epidemic within 2-3 days of infection.

                                **Things to do after the disease occurs**                             
                                1. Irrigation should be stopped until the disease is controlled in the affected land.                             
                                2. As soon as the disease is seen in your own or neighboring fields, spray any approved fungicide or mixture from the following groups at a rate of 2 grams per liter of water and soak the plants well after 4/5 days:                             
                                3. Acrobat M Z (Mancozeb 60% + Dimethomorph 9%) or                             
                                4. Carjet M 8 (Mancozeb 64% + Cymoxanil 8%) or                             
                                5. Secure 600 WG (Mancozeb 5'% +' Fenamidon 10%) or                             
                                6. Melody Duo 66.8 WP (Propineb 70% + Iprovelicarb) 4 grams + Secure 600 WG 1 gram.
                             """)
                    elif(class_name  [result_index] == "Potato___healthy"):
                        st.write(""" 
                                🌱 "Healthy potato plants require proper care! Use quality seeds, apply balanced fertilizer, water regularly, and take effective measures to prevent diseases. Careful cultivation increases yields." 🌱
                             """)
                    elif(class_name  [result_index] == "Rice___Bacterialblight"):
                        st.write(""" 
                                **Rice Bacterial blight** is a harmful disease caused by the bacteria **Xanthomonas oryzae pv. oryzae**. ​​This disease mainly attacks the leaves of rice and reduces the photosynthesis capacity of the plant, resulting in a significant reduction in yield. The use of balanced fertilizers is important in disease prevention and management.
                                                          
                                **Suppression management**                             
                                1. Avoid injury during transplanting.                             
                                2. Do not cultivate susceptible varieties.                             
                                3. Do not use urea fertilizer excessively.                             
                                4. If the disease occurs, water the soil and water again after 7-10 days.                             
                                5. If the disease occurs in the seedling stage, remove the affected seedlings and bring a cushion from the neighboring trees and plant it. The amount of damage will be reduced.                             
                                6. After harvesting the rice, rake the land and burn the stubble.                             
                                7. Spray 50-100 grams of potash fertilizer mixed with 10 liters of water on the affected land in the afternoon.                             
                                8. Apply 5/6 kg of potash fertilizer per bigha as a top dressing.                             
                                9. Do not use urea fertilizer immediately after the Kushikarai storm.                             
                                10. Spray Kupravit 4 grams mixed with Champion 2 grams per liter of water. If necessary, 1% Bordomixture can be used.
                             """)
                    elif(class_name  [result_index] == "Rice___Blast"):
                        st.write(""" 
                                **Rice Blast Disease** is a serious disease caused by a fungus called **Magnaporthe oryzae**. ​​It attacks the leaves, nodes, and ears of the plant, resulting in a significant reduction in rice yield. Balanced use of fertilizers is very important to prevent and control the disease and keep the rice plant strong.

                                **Symptoms of Blast Disease**                             
                                Leaf Blast- Initially, small dark brown spots appear on the affected leaves. Gradually, the spots enlarge and become gray or white in the middle and brown at the edges. The spots are slightly elongated and look like eyes. Multiple spots may merge and eventually the entire leaf may dry out and die.

                                **Things to do to control Blast Disease**                             
                                1. If blast disease occurs, the land should be kept moist. Blast disease is more common in dry land.                             
                                2. If leaf blast disease occurs, an additional 5 kg of potash fertilizer should be applied per bigha of land and the application of urea fertilizer should be stopped.                             
                                3. In the initial stage of leaf blast disease, it is possible to successfully control the disease by applying fungicides similar to those of ear blast disease twice in the late afternoon at intervals of 5-7 days.                             
                                4. There is no chance of controlling the ear blast disease after it has occurred. Therefore, if the disease-friendly environment such as drizzle, hot during the day and cold at night, long mornings wet with dew, cloudy skies, stormy weather prevails, whether the disease is present in the paddy field or not, once when the ear bursts and the ear comes out and again 5-7 days later, 54 grams of Trooper 75WP/ Difa 75WP/ Zil 75WP or 33 grams of Nativo 75 WG, or an approved fungicide of the Tricyclazole/Stubin group should be mixed well in 67 liters of water and sprayed in the late afternoon. It should be remembered that to control the ear blast disease, fungicides must be applied before the disease occurs.
                             """)
                    elif(class_name  [result_index] == "Rice___Brownspot"):
                        st.write(""" 
                                **Rice Brown Spot** is a common disease of rice in our country. Due to errors in seed storage, most farmers are more or less affected by this disease. This disease can occur at any age from seedling stage to the onset of rice milk. However, the incidence of the disease is more common when the seedling is older. This disease is caused by the attack of a fungus called **Bipolaris oryzae**. ​​The level of the disease increases if there is a lack of nutrients or water in the soil. Again, the application of excessive urea can also cause the infection of this disease.

                                **Management: **                             
                                1. Provide moderate irrigation to the seedbed or land.
                                2. Use adequate nitrogen and potash fertilizers in the land.
                                3. This disease cannot increase further if urea is applied in excess.                             
                                4. If the disease is severe, spray with fungicides from the carbendazim group such as: Emcozim 50 WP 1 gm or Noin 50 WP 2 gm or fungicides from the hexaconazole group such as: Contaf 5 EC or Sabab 5 EC at the rate of 1 ml per liter of water. Or spray with other approved fungicides from this group at the approved dosage.
                             """)
                    elif(class_name  [result_index] == "Rice___Healthy"):
                        st.write(""" 
                                🌾Proper care, proper methods, and timely application of fertilizers are very important in healthy rice cultivation. Keep the soil nutrient-rich, provide regular care to protect it from diseases and pests, and ensure the right climate for healthy rice growth. Increase the resistance of rice by using the right fertilizers and ensure good yields. Success in rice cultivation, in proper cultivation!🌾
                             """)
                    elif(class_name  [result_index] == "Rice___Tungro"):
                        st.write(""" 
                                **Rice Tungro disease** is a virus disease caused by **Rice Tungro Bacilliform Virus (RTBV) and Rice Tungro Spherical Virus (RTSV)**. The main vector of this disease is the green leafhopper, which infects the leaves of rice. Initially, Tungro disease attacks a few plants sporadically, but gradually the number of affected plants increases. Sometimes, yellow seedlings are seen in the seedbed due to Tungro disease. Even if there is a nitrogen deficiency, the seedlings turn yellow, so sometimes it is not possible to identify the seedlings affected by Tungro disease.

                                **Things to do to control the disease**                                       
                                1. Regularly inspect the seedbed and the field to monitor the presence of the vector insect (green leafhopper). If the vector insect is present, it should be killed with the help of hand nets or light traps.                             
                                2. Do not make seed beds around the tungro-infected land and if there are abandoned rice plants or weeds in the surrounding land while making seed beds, they should be removed and destroyed.                             
                                3. If a green leafhopper is found in the seed bed of the affected area by pulling the hand net and there are rice plants or host plants infected with tungro nearby, then the following or approved insecticides should be applied to the seed bed twice (10 to 15 days after sowing the seeds and 5 to 7 days before transplanting the seedlings).
                                If symptoms appear sporadically in one or two plants in the field after transplanting the seedlings, the affected plants should be removed and buried in the ground and if there is a carrier insect in the field, immediately apply insecticides such as Mipsin or Sapsin 150 grams or Sevin 227 grams mixed in 6-7 liters of water per bigha.
                             """)
                    elif(class_name  [result_index] == "Tomato___Bacterial_spot"):
                        st.write(""" 
                                **Tomato Bacterial spot** various problems during the winter season due to bacterial diseases. This disease in tomatoes is caused by **Xanthomonas campestris pv.**. It is caused by **Vesicoria** bacteria. Symptoms of this disease include the development of bacterial spots on seedlings and mature plants. In seedlings, infection can cause severe damage.

                                **Preventive measures**                             
                                1. Use approved pest-free seeds if possible.
                                2. Use resistant varieties if available locally.
                                3. Inspect the field regularly, especially during cloudy weather.
                                4. Identify and remove or burn seedlings with leaf spots.
                                5. Keep weeds in and around the field clean.
                                6. Use mulch to prevent soil contamination of the plants.
                                7. Avoid overhead irrigation and avoid working in the field when the leaves are wet.
                                8. After harvesting, plow the crop residue deep into the soil or remove the plant debris.
                                9. Alternatively, plan to leave the land fallow for a few weeks or months after harvesting and dry it with sunlight.
                             """)
                    elif(class_name  [result_index] == "Tomato___Early_blight"):
                        st.write(""" 
                                **Tomato early blight** is a fungal disease caused by **Alternaria solani**. This disease is spread through seeds. As a result, seedlings are infected. Apart from this, this disease can also be spread from other infected plants. In the initial stage, the disease attacks seedlings and later attacks adult plants.

                                **Pest management**
                                1. Seeds should be collected from disease-free plants.                             
                                2. Disease-resistant varieties should be used.                             
                                3. Seeds should be treated by mixing 2 grams of iprodione or mancozeb group fungicides such as Rovral or Dithane M 45 in every liter of water.                             
                                4. Tomatoes should not be cultivated repeatedly in the same land.                             
                                5. Healthy, strong and disease-tolerant varieties should be cultivated.                             
                                6. As soon as 1-2 spots are seen on the leaves, spray 5 teaspoons (20 grams) of Rovral/mancozeb group fungicides in every machine (10 liters) of water and soak the plants well. The second spray should be done 7 days after the first spray. The third spray should be done 15 days after the second spray.                             
                                7. The affected plants should be collected and burned.  
                             """)
                    elif(class_name  [result_index] == "Tomato___Late_blight"):
                        st.write(""" 
                                **Tomato Late Blight** is caused by a pathogen called **Phytophthora infestans**. The fungus spreads to healthy plants through the abandoned parts of diseased tomato plants, infected seeds, wind, rain and irrigation water. If the sky is cloudy or foggy for more than 7-10 days and drizzles, then the chances of this disease in tomato fields are high.                             

                                **Pest Management**                             
                                1. Healthy seeds should be collected from disease-free areas and healthy seedlings should be planted.                             
                                2. If the sky is cloudy or foggy and drizzles for more than 3-4 days, without delay, mix 5 teaspoons (20 grams) of Mancozeb group fungicide per machine (10 liters) of water and spray 2 times 3 days after the first spray. Then spray 3 times after 7 days.                             
                                3. The affected crops should be completely burned and destroyed.                             
                                4. Fertilizer and irrigation should be applied in moderation and on time.  
                             """)
                    elif(class_name  [result_index] == "Tomato___healthy"):
                        st.write(""" 
                                 Proper care and maintenance are important for healthy tomato cultivation! Ensure regular irrigation, use of disease resistant varieties, balanced fertilizer application and timely fungicide application. Give your tomato plants enough sun and prepare the land well. Take care of your tomato plants, the yield will be high and the quality will be the best. Move forward for successful cultivation, good health and delicious tomatoes!🍅
                             """)
                    elif(class_name  [result_index] == "Tomato___Leaf_Mold"):
                        st.write(""" 
                                 **Tomato Leaf Mold** is caused by a fungus called Passalora fulva. It usually spreads rapidly in hot summers and humid environments.

                                 **Management**                             
                                 1. Sprinkler irrigation instead of flood irrigation.
                                 2. Remove affected fruits, leaves and tips.
                                 3. Prepare the land by deep cultivation before planting seeds.                            
                                 4. Spray propiconazole group fungicides such as Tilt 250 EC  at the rate of 0.5 ml. / l. in water 3 times in a row for 10 days in the late afternoon.
                             """)
                    elif(class_name  [result_index] == "Tomato___Septoria_leaf_spot"):
                        st.write(""" 
                            Septoria leaf spot of tomato occurs worldwide, caused by a fungus called **Septoria lycopersici**. This fungus only affects potatoes and tomatoes. The optimum temperature for fungal growth is 15° to 27° Celsius, but fungal growth is maximum at 25° Celsius.

                            **Preventive measures**                             
                            1. Collect disease-free certified seeds.
                            2. Use disease-resistant varieties if possible.
                            3. Use organic or plastic-based mulch to prevent soil contamination.
                            4. Remove and destroy affected leaves.
                            5. Increase air circulation by erecting plants with stakes.
                            6. Prevent plants from falling over.
                            7. Ensure that the crop field is free of weeds that are easily affected.
                            8. Avoid sprinklers or overhead irrigation.
                            9. Remove or destroy plant residues by deep tillage immediately after harvest.
                            10. Follow crop rotation with non-native plants that are not members of the Solanaceae family.
                            """)
                    elif(class_name [result_index] == "Tomato___Spider_mites  Two-spotted_spider_mite"):
                        st.write(""" 
                            **Tomato Spider Mites** are a common problem for crop damage. They suck the sap from the leaves of the plant and reduce the yield of the crop. Various effective repellents or acaricides can be used for this problem.

                            **Acaricides**                             
                            **Abamectin: **0.5-1 ml per liter of water.                             
                            **Bifenazate: **As per the label.                             
                            **Spiromesifen: **1-2 ml per liter of water.                             
                            **Fenpyroximate: **1-2 ml per liter of water.                             
                            **Propargite: **2-3 ml per liter of water.                             
                            **Teflubenzuron: **1-2 ml per liter of water.                             
                            """)
                    elif(class_name  [result_index] == "Tomato___Target_Spot"):
                        st.write(""" 
                            **Tomato Target spot disease** is caused by the fungus **Corynespora cassiicola**. It causes round, target-like spots on the leaves and if the disease is not controlled, the growth and yield of the entire plant can be severely affected.

                            **Symptoms**                             
                            1. Small, round to elliptical dark brown spots appear on the upper and lower leaves.
                            2. Target-like shapes can be seen between the spots.
                            3. The spots may coalesce and the leaves may dry out.
                            4. The plant becomes weak and the yield is reduced.

                            **Target spot disease prevention**                             
                            **Mancozeb:** 2.5-3 grams per liter of water.
                            **Chlorothalonil:** 2-3 grams per liter of water.                             
                            """)
                    elif(class_name  [result_index] == "Tomato___Tomato_Yellow_Leaf_Curl_Virus"):
                        st.write(""" 
                            **Tomato yellow leaf curl virus (TYLCV)** is not a seed-borne disease and is not transmitted by hand-to-hand contact. The disease is spread by whiteflies of the species **Bemisia tabaci**. Whiteflies feed on the underside of most leaves and attack young, tender plants.

                            **Insecticides for prevention**                             
                             **Imidacloprid:** 1 ml per liter of water.                             
                            **Thiamethoxam:** 0.3-0.5 g per liter of water.                             
                            **Acetamiprid:** 1 g per liter of water.                             
                            **Spiromesifen:** 1-2 ml per liter of water.                             
                            """)
                    elif(class_name  [result_index] == "Tomato___Tomato_mosaic_virus"):
                        st.write(""" 
                            **Tomato Tomato mosaic survive** in dry soil for more than 2 years in plant or root debris. The virus enters the plant through small wounds on the roots and can be spread through infected seeds, seedlings, weeds, and infected plant parts. The virus can also be spread from field to field by wind, rain, grasshoppers, small mammals, and birds. The infection can also spread from field to field if the intercropping of the crop is poor.

                            **Preventive measures**
                            1. Use steam-pasteurization to disinfect the soil in the seedbed.
                            2. Do not plant seedlings in fields previously infected with the virus.
                            3. Intercropping the crop by washing hands, wearing gloves, and disinfecting cultivation tools and equipment.
                            4. Monitor the seedbed and field, remove and burn diseased plants.
                            5. Identify and eliminate weeds in and around the field.
                            6. After harvesting, hoe the soil and bury or burn the plant residues.
                            7. Avoid planting companion plants that provide shelter near tomato fields.                             
                            """)
                    else:
                        st.write("Not found disease name")

                except TypeError as e:
                    st.error(f"Try again with another image.")

    except Exception as e:
        st.error(f"please select Browse files and Choose image")
