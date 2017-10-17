from PIL import Image, ImageFilter
import os
count =0
for filename in os.listdir("./train_images"):
    
    #Read image
    try:
        im = Image.open( "./train_images/"+filename )

        #Splitting the image into its respective bands, i.e. Red, Green,
        #and Blue for RGB
        r,g,b = im.split()
        count = count +1
    except:
        print("BAD: " + filename)
        os.remove("./train_images/"+filename)

    if count%1000 == 0 :
        print(str(count))
