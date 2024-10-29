import sys
sys.path.append('../')
#from utils import get_center_of_bbox, measure_distance

def get_center_of_bbox(bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)
    
def measure_distance(p1, p2):
       return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)**0.5


class playerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70
    
    


    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = 9999
        assigned_palyers = -1
        
        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_palyers = player_id
        return assigned_palyers