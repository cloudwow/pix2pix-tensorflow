from PIL import Image, ImageFilter
import os
count =0
for filename in os.listdir("./simpsons"):
    
    #Read image
    try:
        im = Image.open( "./simpsons/"+filename )

        #Splitting the image into its respective bands, i.e. Red, Green,
        #and Blue for RGB
        r,g,b = im.split()
        count = count +1
    except:
        print("BAD: " + filename)

    if count%1000 == 0 :
        print(str(count))
