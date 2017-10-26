import sys
import threading
import urllib
import urllib.request
def grab_screenshot(destdir, season, episode, image_index):
    image_name = destdir+"/s{season:02d}e{episode:02d}_{image_index}.jpg".format(**locals())
    image_url = "https://s3.amazonaws.com/images.springfieldspringfield.co.uk/screencaps/south-park/season{season}/s{season:02d}e{episode:02d}/s{season:02d}e{episode:02d}_{image_index}.jpg".format(**locals())
    # image = urllib.URLopener()
    urllib.request.urlretrieve(image_url, image_name)
    print(image_url)

def grab_season(destdir, season):
    for episode in range(1,18):
        for image_index in range(2,290):
            grab_screenshot(destdir, season, episode, image_index)

destdir=sys.argv[1]
print("dest dir: " + str(destdir))
threads = []
for season in range(2,24):
    t = threading.Thread(target=grab_season, args=(destdir, season))
    threads.append(t)
    t.start()
