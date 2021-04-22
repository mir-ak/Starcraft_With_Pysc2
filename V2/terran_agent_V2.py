from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import units
from absl import app
import numpy as np
import random

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_TRAIN_MARINE_QUICK = actions.FUNCTIONS.Train_Marine_quick.id
_TRAIN_HELLION_QUICK = actions.FUNCTIONS.Train_Hellion_quick.id
_BUILD_SUPPLYDEPOT_SCREEN = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS_SCREEN = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_REFINERY_SCREEN = actions.FUNCTIONS.Build_Refinery_screen.id
_BUILD_FACTORY_SCREEN = actions.FUNCTIONS.Build_Factory_screen.id
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PATROL_MINIMAP = 334
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
    factory_top = [53, 70]
    factory_bottom = [31, 12]
    barracks_top = [[75, 30], [75, 45], [75, 62]]
    barracks_buttom = [[9, 20], [9, 40], [9, 50]]
    supply_depots_top = [[65, 30], [65, 40], [65, 50], [65, 60]]
    supply_depots_buttom = [[20, 23], [20, 33], [20, 43], [20, 53]]
    patrol_point_top = [[10, 20],[45,20],[10, 20],[45,20]]
    patrol_point_buttom = [[60, 65],[10,65],[60, 65],[10,65]]
    prev_total_value_structures = 400
    top_pos = None
    structures_ok = None

    def reset(self):
        super(SimpleAgent, self).reset()
        self.state = 0
        self.state_m = 0
        self.tmp = [True, False]
        self.state_one = 0
        self.index_b = 0
        self.index_sd = 0
        self.index_ry = 0
        self.index_pt = 0
        self.refinery_top = [[58,21],[23,56]]
        self.refinery_bottom = [[25,62],[60,27]]
        self.barracks_top = [[75, 30], [75, 45], [75, 62]]
        self.barracks_buttom = [[9, 20], [9, 36], [9, 55]]
        self.supply_depots_top = [[65, 30], [65, 40], [65, 50], [65, 60]]
        self.supply_depots_buttom = [[20, 23], [20, 33], [20, 43], [20, 53]]
        self.factory_top = [53, 70]
        self.factory_bottom = [31, 12]
        self.patrol_point_top = [[10, 20],[45,20],[10, 20],[45,20]]
        self.patrol_point_buttom = [[60, 65],[10,65],[60, 65],[10,65]]
        self.prev_total_value_structures = 400
        self.top_pos = None
        self.structures_ok = None
        self.state_attack = 0
    
    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]
    
     # creation des marines 
    def train_MARINES(self, obs):
        if self.state_m == 0:
            target = [0, 0]
            if self.top_pos == False:
                target = self.barracks_buttom[0]
            if self.top_pos == True:
                target = self.barracks_top[0]
            self.state_m = 1
            return actions.FUNCTIONS.select_point("select_all_type", target)
        if self.state_m == 1:
            unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
            marine_y, marine_x = (unit_type == units.Terran.Marine).nonzero()
            if len(marine_y) > 160:
                self.state_m = 2
            if _TRAIN_MARINE_QUICK in obs.observation["available_actions"]:
                return actions.FUNCTIONS.Train_Marine_quick("now")
            return actions.FUNCTIONS.no_op()
        if self.state_m == 2:
            return self.attack_with_patrol(obs)
        return actions.FUNCTIONS.no_op()
    
    # creation des Hellions pour factory mais il fonction pas bien 
    """  def train_Hellions(self, obs):
            if self.state_m == 0:
                target = [0, 0]
                if self.top_pos == False:
                    target = self.factory_bottom[0]
                if self.top_pos == True:
                    target = self.factory_top[0]
                self.state_m = 1
                return actions.FUNCTIONS.select_point("select_all_type", target)
            if self.state_m == 1:
                unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
                hellion_y, hellion_x = (unit_type == units.Terran.Hellion).nonzero()
                if len(hellion_y) > 3:
                    self.state_m = 2
                if _TRAIN_HELLION_QUICK in obs.observation["available_actions"]:
                    return actions.FUNCTIONS.Train_Hellion_quick("now")
                return actions.FUNCTIONS.no_op()
            if self.state_m == 2:
                return self.attack_with_patrol(obs)
            return actions.FUNCTIONS.no_op()"""
            
    # selction un scv
    def Select_One_SCV(self, obs):
        unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
        scv_y, scv_x = (unit_type == units.Terran.SCV).nonzero()
        index = random.randrange(0, 12)
        if not scv_x.any():
            return actions.FUNCTIONS.no_op()
        target = [scv_x[index], scv_y[index]]
        if _SELECT_POINT in obs.observation["available_actions"]:
            return actions.FunctionCall(_SELECT_POINT, [_SCREEN, target])
        return actions.FUNCTIONS.no_op()
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
            elif (type_build == 3):
                target = self.factory_top
        else:
            if (type_build == 0):
                target = self.supply_depots_buttom[self.index_sd]
                self.index_sd = self.index_sd + 1
            elif (type_build == 1):
                target = self.barracks_buttom[self.index_b]
                self.index_b = self.index_b + 1
            elif (type_build == 2):
                target = self.refinery_bottom[self.index_ry]
                self.index_ry = self.index_ry + 1
            elif (type_build == 3):
                target = self.factory_bottom
              
        return target

    # construire BTM
    def created_supply_depot_and_barracks_and_refinery_and_factory(self, obs):
        total_value_structures = obs.observation["score_cumulative"]["total_value_structures"]
        if self.prev_total_value_structures != total_value_structures:
            return actions.FUNCTIONS.no_op()
        if self.state == 0:
            self.state = 1
            return self.Select_One_SCV(obs)
        #construire supply_depots
        if self.state == 1:
            if _BUILD_SUPPLYDEPOT_SCREEN in obs.observation["available_actions"] and self.prev_total_value_structures < 800:
                target = self.generate_coord(obs, 0)
                self.prev_total_value_structures = total_value_structures + 100
                if self.prev_total_value_structures == 800:
                    self.state = 2
                return actions.FunctionCall(_BUILD_SUPPLYDEPOT_SCREEN, [_NOT_QUEUED, target])
        #construire refinery
        if self.state == 2 :
            #select un autre scv pour le garde dans la refinery 1 pour recolte 
            if self.tmp[0]:
                self.tmp[0] = False
                return self.Select_One_SCV(obs)
            #select un autre scv pour le garde dans la refinery 2 pour recolte 
            if self.tmp[1]:
                self.tmp[1] = False
                return self.Select_One_SCV(obs)
            if _BUILD_REFINERY_SCREEN in obs.observation["available_actions"]:
                target = self.generate_coord(obs, 2)
                self.prev_total_value_structures = total_value_structures + 75
                self.tmp[1] = True
                if self.prev_total_value_structures == 950:
                    self.state = 3
                return actions.FunctionCall(_BUILD_REFINERY_SCREEN, [_NOT_QUEUED, target])
        #construire barrackas
        if self.state == 3:
            if self.tmp[1]:
                self.tmp[1] = False
                return self.Select_One_SCV(obs)
            if _BUILD_BARRACKS_SCREEN in obs.observation["available_actions"]:
                target = self.generate_coord(obs, 1)
                self.prev_total_value_structures = total_value_structures + 150
                if self.prev_total_value_structures == 1400:
                    self.state = 4
                return actions.FunctionCall(_BUILD_BARRACKS_SCREEN, [_NOT_QUEUED, target])
        #construire factory
        if self.state == 4 :
            if _BUILD_FACTORY_SCREEN in obs.observation["available_actions"]:
                target = self.generate_coord(obs, 3)
                self.prev_total_value_structures = total_value_structures + 250
                return actions.FunctionCall(_BUILD_FACTORY_SCREEN, [_NOT_QUEUED, target])
        return actions.FUNCTIONS.no_op()
        
    # la fonction qui assure l'envoi des marines en patrouille
    def attack_with_patrol(self, obs):
        if self.state_attack == 0:
            unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
            marine_y, marine_x = (unit_type == units.Terran.Marine).nonzero()
            if not marine_x.any():
                return actions.FUNCTIONS.no_op()
            target = (marine_x[0], marine_y[0])
            self.state_attack = 1
            self.state_m = 2
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
                self.state_m = 0
                self.state_attack = 0
                return actions.FunctionCall(_PATROL_MINIMAP, [_NOT_QUEUED, target])
            return actions.FUNCTIONS.no_op()
        return actions.FUNCTIONS.no_op()
    
    def step(self, obs):
        super(SimpleAgent, self).step(obs)
        total_value_structures = obs.observation["score_cumulative"]["total_value_structures"]
        collected_minerals = obs.observation["score_cumulative"]["collected_minerals"]
        if (self.top_pos == None):
            player_y, player_x = (
                obs.observation["feature_minimap"][_PLAYER_RELATIVE] == 1).nonzero()
            self.top_pos = player_y.mean() <= 31
        if (total_value_structures == 1650) and (self.structures_ok == None):
            self.state_one = 1  
            self.structures_ok = True
            return actions.FUNCTIONS.no_op()
        if collected_minerals < 50:
            return actions.FUNCTIONS.no_op()
        if self.state_one == 0:
            return self.created_supply_depot_and_barracks_and_refinery_and_factory(obs)
        if self.state_one == 1:
            return self.train_MARINES(obs)
        return actions.FUNCTIONS.no_op()
def main(unused_argv):
    agent = SimpleAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.terran),
                        sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)],
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
