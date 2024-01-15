szybka instrukcja:
-jak chcesz zamiast nagrania miec kamere na zywo zmieniasz camera na to zakomentowane: 

# camera object
camera = cv.VideoCapture("fast.mp4")
# camera = cv.VideoCapture(0)

-jest dodana funcja kalibracji. dziala to tak, ze jak włączy się kamera to najpierw patrzysz w lewo, klkasz c, patrzysz w prawo, klikasz c, potem gora i dol.
i punkty wychylenia twojej zrenicy zostaja zapisane i jak spojrzysz dalej niz to co zaznacyles pokaze sie kierunek patrzenia. Poprobuj <3
