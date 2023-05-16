import cv2

from ModuleDisplayScreen import ResizeImage

class Floorplan:
    url = ''

    roomsCoord = {
        'zaal_A': [823, 999],
        'zaal_B': [639, 997],
        'zaal_C': [459, 997],
        'zaal_D': [499, 837],
        'zaal_E': [775, 841],
        'zaal_F': [835, 669],
        'zaal_G': [633, 669],
        'zaal_H': [463, 669],
        'zaal_I': [695, 535],
        'zaal_J': [735, 417],
        'zaal_K': [815, 313],
        'zaal_L': [933, 245],
        'zaal_M': [453, 461],
        'zaal_N': [241, 535],
        'zaal_O': [57, 535],
        'zaal_P': [459, 243],
        'zaal_Q': [609, 243],
        'zaal_R': [461, 71],
        'zaal_S': [743, 79],
        'zaal_1': [1267, 1000],
        'zaal_2': [1440, 1000],
        'zaal_3': [1630, 1000],
        'zaal_4': [1630, 840],
        'zaal_5': [1314, 840],
        'zaal_6': [1267, 674],
        'zaal_7': [1440, 674],
        'zaal_8': [1630, 674],
        'zaal_9': [1377, 530],
        'zaal_10': [1350, 414],
        'zaal_11': [1267, 306],
        'zaal_12': [1155, 244],
        'zaal_13': [1637, 457],
        'zaal_14': [1839, 537],
        'zaal_15': [2037, 540],
        'zaal_16': [1639, 232],
        'zaal_17': [1475, 232],
        'zaal_18': [1633, 85],
        'zaal_19': [1316, 85],
        'zaal_V': [1046, 1144],
        'zaal_II': [1046, 835],
        'zaal_III': [1046, 478],
    }


    def __init__(self, url, debug=False):
        self.url = url + '\\msk_floorplan.png'


    def DrawPath(self, roomSequence):
        img = cv2.imread(self.url)
        
        previousRoom = ''
        for room in roomSequence:
            img = cv2.circle(img, self.roomsCoord[room], 10, [255, 0, 0], 10)
            if previousRoom != '':
                img = cv2.line(img, self.roomsCoord[room], self.roomsCoord[previousRoom], [0, 0, 0], 3)
            previousRoom = room
        img = cv2.circle(img, self.roomsCoord[roomSequence[-1]], 10, [0, 255, 0], 10)
        img = ResizeImage(img)

        return img
    

    def DrawRoom(self, room):
        room = room[0].lower() + room[1:]
        img = cv2.imread(self.url)
        img = cv2.circle(img, self.roomsCoord[room], 10, [0, 255, 0], 10)
        img = ResizeImage(img)

        return img