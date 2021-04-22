
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import units
from absl import app
import numpy as np
import random
# Functions
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_TRAIN_MARINE_QUICK = actions.FUNCTIONS.Train_Marine_quick.id
_TRAIN_HELLION_QUICK = actions.FUNCTIONS.Train_Hellion_quick.id
_BUILD_SUPPLYDEPOT_SCREEN = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS_SCREEN = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_REFINERY_SCREEN = actions.FUNCTIONS.Build_Refinery_screen.id
_BUILD_FACTORY_SCREEN = actions.FUNCTIONS.Build_Factory_screen.id
# Features
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PATROL_MINIMAP = 334
# Parameters
_NOT_QUEUED = [0]
_SCREEN = [0]


class SimpleAgent(base_agent.BaseAgent):
    state = 0
    state_m = 0
    tmp = 0 
    state_one = 0
    index_b = 0
    index_sd = 0
    index_ry = 0
    index_pt = 0
    refinery_top = [[58,21],[23,56]]
    refinery_bottom = [[25,62],[60,27]]
    factory_top = [40, 62]
    factory_bottom = [37, 20]
    barracks_top = [[75, 30], [75, 45], [75, 62]]
    barracks_buttom = [[9, 20], [9, 40], [9, 50]]
    supply_depots_top = [[65, 30], [65, 40], [65, 50], [65, 60]]
    supply_depots_buttom = [[20, 23], [20, 33], [20, 43], [20, 53]]
    patrol_point_top = [10, 10]
    patrol_point_top = [[10, 20],[45,20],[10, 20],[45,20]]
    patrol_point_buttom = [[60, 65],[10,65],[60, 65],[10,65]]
    prev_total_value_structures = 400
    top_pos = None
    structures_ok = None

    def reset(self):
        super(SimpleAgent, self).reset()
        # state controller les etats de construction
        self.state = 0
        #controller les etats de attaque
        self.state_m = 0
        #controller les etats de jeu entre construction et envoi des marines
        self.state_one = 0
        #index des barrack a dessiner
        self.index_b = 0
        #index des supply depots a dessiner 
        self.index_sd = 0
        #index des refinery depots a dessiner 
        self.index_ry = 0
        #index des cord attaque a patrouiller
        self.index_pt = 0
        #les cord refinery depots ou cas ou la base est en haut a gauche
        self.refinery_top = [[58,21],[23,56]]
         #les cord refinery depots ou cas ou la base est en bas a droite
        self.refinery_bottom = [[25,62],[60,27]]
        #les cord barrack ou cas ou la base est en haut a gauche
        self.barracks_top = [[75, 30], [75, 45], [75, 62]]
        #les cord barrack ou cas ou la base est en bas a droite
        self.barracks_buttom = [[9, 20], [9, 36], [9, 55]]
        #les cord supply depots ou cas ou la base est en haut a gauche
        self.supply_depots_top = [[65, 30], [65, 40], [65, 50], [65, 60]]
        #les cord supply depots ou cas ou la base est en bas a droite
        self.supply_depots_buttom = [[20, 23], [20, 33], [20, 43], [20, 53]]
        #les cord factory depots ou cas ou la base est en haut a gauche
        self.factory_top = [40, 62]
        #les cord factory depots ou cas ou la base est en bas a droite
        self.factory_bottom = [37, 20]
        #les points vers lequelle patrouiller quand la base est en bas
        self.patrol_point_top = [[10, 20],[45,20],[10, 20],[45,20]]
        #le point vers lequelle patrouiller quand la base est en haut
        self.patrol_point_buttom = [[60, 65],[10,65],[60, 65],[10,65]]
        self.prev_total_value_structures = 400
        #determiner la base si c est en bas ou en haut
        self.top_pos = None
        #savoir si tout est bien construit
        self.structures_ok = None
        #depart des marines
        self.state_attack = 0
        
    # selction un scv
    def select_scv_tst(self, obs):
        scvs = self.get_units_by_type(obs, units.Terran.SCV)
        if len(scvs) > 0:
            scv = random.choice(scvs)
            return actions.FUNCTIONS.select_point("select_all_type", (scv.x, scv.y))
        
    # creation des marines 
    def train_MARINES(self, obs):
        if self.state == 0:
            target = [0, 0]
            if self.top_pos == False:
                target = self.barracks_buttom[0]
            if self.top_pos == True:
                target = self.barracks_top[0]
            self.state = 1
            return actions.FUNCTIONS.select_point("select_all_type", target)
        if self.state == 1:
            unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
            marine_y, marine_x = (unit_type == units.Terran.Marine).nonzero()
            if len(marine_y) > 160:
                self.state = 2
            if _TRAIN_MARINE_QUICK in obs.observation["available_actions"]:
                return actions.FUNCTIONS.Train_Marine_quick("now")
            return actions.FUNCTIONS.no_op()
        if self.state == 2:
            return self.attack_with_patrol(obs)
        return actions.FUNCTIONS.no_op()
        
    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]
    #gérer les coordonnees en fontion de type de BTM et de l'emplacement de la base dans la minimap
    def generate_coord(self, obs, type_build):
        target = [0, 0]
        if (self.top_pos == True):
            if (type_build == 0):
                target = self.supply_depots_top[self.index_sd]
                self.index_sd = self.index_sd + 1
            elif (type_build == 1):
                target = self.barracks_top[self.index_b]
                self.index_b = self.index_b + 1
            elif (type_build == 2):
                target = self.refinery_top[self.index_ry]
                self.index_ry = self.index_ry +1 
        else:
            if (type_build == 0):
                target = self.supply_depots_buttom[self.index_sd]
                self.index_sd = self.index_sd + 1
            elif (type_build == 1):
                target = self.barracks_buttom[self.index_b]
                self.index_b = self.index_b + 1
            elif (type_build == 2):
                target = self.refinery_bottom[self.index_ry]
                self.index_ry = self.index_ry +1 
        return target

    # construire BTM(suply et barracks et refinery si vous mettez a la ligne 234 le 1250 a 1400)
    def created_supply_depot_and_barracks(self, obs):
        total_value_structures = obs.observation["score_cumulative"]["total_value_structures"]
        if self.prev_total_value_structures != total_value_structures:
            return actions.FUNCTIONS.no_op()
        if self.state == 0:
            self.state = 1
            return self.select_scv_tst(obs)
        #construire les supply_depots
        if self.state == 1:
            if _BUILD_SUPPLYDEPOT_SCREEN in obs.observation["available_actions"] and self.prev_total_value_structures < 800:
                target = self.generate_coord(obs, 0)
                self.prev_total_value_structures = total_value_structures + 100
                if self.prev_total_value_structures == 800:
                    self.state = 2
                return actions.FunctionCall(_BUILD_SUPPLYDEPOT_SCREEN, [_NOT_QUEUED, target])
        #construire les barracks
        if self.state == 2:
            if _BUILD_BARRACKS_SCREEN in obs.observation["available_actions"]:
                target = self.generate_coord(obs, 1)
                self.prev_total_value_structures = total_value_structures + 150
                if self.prev_total_value_structures == 1250:
                    self.state = 3
                return actions.FunctionCall(_BUILD_BARRACKS_SCREEN, [_NOT_QUEUED, target])
        #construire les refinery
        if self.state == 3 :
            if _BUILD_REFINERY_SCREEN in obs.observation["available_actions"]:
                target = self.generate_coord(obs, 2)
                self.prev_total_value_structures = total_value_structures + 75
                return actions.FunctionCall(_BUILD_REFINERY_SCREEN, [_NOT_QUEUED, target])
        return actions.FUNCTIONS.no_op()
    
    #la fonction qui envoi des marines en patrouille
    def attack_with_patrol(self, obs):
        if self.state_attack == 0:
            unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
            marine_y, marine_x = (unit_type == units.Terran.Marine).nonzero()
            if not marine_x.any():
                return actions.FUNCTIONS.no_op()
            target = (marine_x[0], marine_y[0])
            self.state_attack = 1
            self.state = 2
            return actions.FUNCTIONS.select_army("select", target)
        if self.state_attack == 1:
            if _PATROL_MINIMAP in obs.observation["available_actions"]:
                if self.top_pos == False:
                    target = self.patrol_point_top[self.index_pt]
                    self.index_pt = self.index_pt + 1
                    if self.index_pt == 4:
                        target = self.patrol_point_top[0]
                else:
                    target = self.patrol_point_buttom[self.index_pt]
                    self.index_pt = self.index_pt + 1
                    if self.index_pt == 4:
                        target = self.patrol_point_buttom[0]
                self.state = 0
                self.state_attack = 0
                return actions.FunctionCall(_PATROL_MINIMAP, [_NOT_QUEUED, target])
            return actions.FUNCTIONS.no_op()
        return actions.FUNCTIONS.no_op()

    def step(self, obs):
        super(SimpleAgent, self).step(obs)
        total_value_structures = obs.observation["score_cumulative"]["total_value_structures"]
        collected_minerals = obs.observation["score_cumulative"]["collected_minerals"]
        # determiner la position de la base
        if (self.top_pos == None):
            player_y, player_x = (
                obs.observation["feature_minimap"][_PLAYER_RELATIVE] == 1).nonzero()
            self.top_pos = player_y.mean() <= 31
        #si vous voulez cree les refinry vous mettez 1250 a 1400 mais ca réduit la vitesse de la création des marine 
        if (total_value_structures == 1250) and (self.structures_ok == None):
            self.state = 0
            self.state_one = 1
            self.structures_ok = True
            return actions.FUNCTIONS.no_op()
        # ne rien faire tant que ya pas assez de mineraux 
        if collected_minerals < 50:
            return actions.FUNCTIONS.no_op()
        if self.state_one == 0:
            return self.created_supply_depot_and_barracks(obs)
        if self.state_one == 1:
            return self.train_MARINES(obs)
        return actions.FUNCTIONS.no_op()

# main 
def main(unused_argv):
    agent = SimpleAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.terran),
                        sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.easy)],
                agent_interface_format=features.AgentInterfaceFormat( 
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    use_feature_units=True
            ),
                game_steps_per_episode=0,
                visualize=True
            ) as env:
                agent.setup(env.observation_spec(), env.action_spec())
                timesteps = env.reset()
                agent.reset()
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    app.run(main)
