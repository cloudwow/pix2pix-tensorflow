import threading
import urllib
import urllib.request
def grab_screenshot(season, episode, image_index):
    image_name = "simpsons/s{season:02d}e{episode:02d}_{image_index}.jpg".format(**locals())
    image_url = "https://s3.amazonaws.com/images.springfieldspringfield.co.uk/screencaps/the-simpsons/season{season}/s{season:02d}e{episode:02d}/s{season:02d}e{episode:02d}_{image_index}.jpg".format(**locals())
    # image = urllib.URLopener()
    urllib.request.urlretrieve(image_url, image_name)
    print(image_url)

def grab_season(season):
    for episode in range(1,22):
        for image_index in range(1,220):
            grab_screenshot(season, episode, image_index)
            
threads = []
for season in range(2,24):
    t = threading.Thread(target=grab_season, args=(season,))
    threads.append(t)
    t.start()
