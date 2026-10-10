import pygame
import random
import math
class Fighter:
    def __init__(self, player, x, y, flip, data, sprite_sheet, animation_steps, sound):
        self.player = player
        self.size = data[0]
        self.image_scale = data[1]
        self.offset = data[2]
        self.flip = flip
        self.animation_list = self.load_images(sprite_sheet, animation_steps)
        self.action = 0  # 0:idle #1:run #2:jump #3:attack1 #4: attack2 #5:hit #6:death
        self.frame_index = 0
        self.image = self.animation_list[self.action][self.frame_index]
        self.update_time = pygame.time.get_ticks()
        self.rect = pygame.Rect((x, y, 80, 180))
        self.vel_y = 0
        self.running = False
        self.jump = False
        self.attacking = False
        self.attack_type = 0
        self.attack_cooldown = 0
        self.attack_sound = sound
        self.hit = False
        self.health = 100
        self.alive = True
        # AI attributes for computer player
        self.ai_enabled = (player == 2)  # Player 2 is AI-controlled
        self.ai_reaction_time = 0
        self.ai_decision_cooldown = 0
        self.ai_aggression = 0.85  # Increased aggression (0-1 scale, higher = more aggressive)
        self.ai_defense_threshold = 25  # Lower threshold for defensive behavior
        self.ai_reaction_speed = 5  # Faster reaction time (lower = faster)
        self.ai_predictive_aim = True  # Enable predictive aiming

    def load_images(self, sprite_sheet, animation_steps):
        # extract images from spritesheet
        animation_list = []
        for y, animation in enumerate(animation_steps):
            temp_img_list = []
            for x in range(animation):
                temp_img = sprite_sheet.subsurface(x * self.size, y * self.size, self.size, self.size)
                temp_img_list.append(
                    pygame.transform.scale(temp_img, (self.size * self.image_scale, self.size * self.image_scale)))
            animation_list.append(temp_img_list)
        return animation_list

    def ai_make_decision(self, target, screen_width, screen_height):
        """Enhanced AI decision making for computer-controlled fighter"""
        if not self.ai_enabled or not self.alive:
            return 0, 0, False, 0  # dx, dy, jump, attack_type
        
        # Reduce decision cooldown
        if self.ai_decision_cooldown > 0:
            self.ai_decision_cooldown -= 1
            return 0, 0, False, 0
        
        # Calculate distance to target
        distance = abs(target.rect.centerx - self.rect.centerx)
        vertical_distance = abs(target.rect.centery - self.rect.centery)
        
        # Predictive aiming - anticipate player movement
        target_velocity_x = 0
        if hasattr(target, 'running') and target.running:
            target_velocity_x = 10 if target.rect.centerx > self.rect.centerx else -10
        
        # AI behavior based on health
        is_defensive = self.health < self.ai_defense_threshold
        
        # Movement decisions
        dx = 0
        dy = 0
        should_jump = False
        attack_type = 0
        
        # Enhanced movement logic
        if distance > 180:  # Too far - approach aggressively
            if target.rect.centerx > self.rect.centerx:
                dx = 12  # Move right faster
            else:
                dx = -12  # Move left faster
            self.running = True
        elif distance < 60:  # Too close - maintain optimal distance
            if target.rect.centerx > self.rect.centerx:
                dx = -8  # Move left
            else:
                dx = 8  # Move right
            self.running = True
        elif distance < 120:  # Optimal range - circle around opponent
            if random.random() < 0.3:  # 30% chance to circle
                if target.rect.centerx > self.rect.centerx:
                    dx = -3
                else:
                    dx = 3
                self.running = True
        
        # Enhanced jump decisions
        jump_chance = 0.4 if not is_defensive else 0.6  # More likely to jump when defensive
        if (random.random() < jump_chance and not self.jump and 
            (distance < 200 or vertical_distance > 50 or target.jump)):
            should_jump = True
        
        # Counter-attack logic - attack when opponent is vulnerable
        should_attack = False
        if distance < 140 and self.attack_cooldown == 0:
            # Attack when opponent is jumping or attacking
            if target.jump or target.attacking:
                should_attack = True
            # Normal attack decision
            elif random.random() < self.ai_aggression:
                should_attack = True
        
        if should_attack:
            # Smart attack type selection
            if target.jump:
                attack_type = 2  # Use long range attack against jumping opponent
            elif distance < 80:
                attack_type = 1  # Close range attack
            else:
                attack_type = 2  # Long range attack
            
            # Faster decision cooldown for more aggressive AI
            self.ai_decision_cooldown = random.randint(5, 15)
        
        # Enhanced defensive behavior
        if is_defensive:
            # Increase distance when health is low
            if distance < 250:
                if target.rect.centerx > self.rect.centerx:
                    dx = -10
                else:
                    dx = 10
                self.running = True
            # More likely to jump to escape
            if random.random() < 0.5 and not self.jump:
                should_jump = True
            # Block/evade more often
            if target.attacking and distance < 100:
                if random.random() < 0.7:  # 70% chance to evade
                    should_jump = True
        
        return dx, dy, should_jump, attack_type

    def move(self, screen_width, screen_height, target, round_over):
        SPEED = 10
        GRAVITY = 2
        dx = 0
        dy = 0
        self.running = False
        self.attack_type = 0

        # get keypresses for human player
        key = pygame.key.get_pressed()

        # can only perform other actions if not currently attacking
        if self.attacking == False and self.alive == True and round_over == False:
            # check player 1 controls (human player)
            if self.player == 1:
                # movement
                if key[pygame.K_a]:
                    dx = -SPEED
                    self.running = True
                if key[pygame.K_d]:
                    dx = SPEED
                    self.running = True
                # jump
                if key[pygame.K_w] and self.jump == False:
                    self.vel_y = -30
                    self.jump = True
                # attack
                if key[pygame.K_r] or key[pygame.K_t]:
                    self.attack(target)
                    # determine which attack type was used
                    if key[pygame.K_r]:
                        self.attack_type = 1
                    if key[pygame.K_t]:
                        self.attack_type = 2

            # check player 2 controls (AI player)
            elif self.player == 2 and self.ai_enabled:
                # Get AI decisions
                ai_dx, ai_dy, should_jump, attack_type = self.ai_make_decision(target, screen_width, screen_height)
                
                # Apply AI movement
                if ai_dx != 0:
                    dx = ai_dx
                    self.running = True
                
                # Apply AI jump
                if should_jump and self.jump == False:
                    self.vel_y = -30
                    self.jump = True
                
                # Apply AI attack
                if attack_type > 0:
                    self.attack(target)
                    self.attack_type = attack_type

        # apply gravity
        self.vel_y += GRAVITY
        dy += self.vel_y

        # ensure player stays on screen
        if self.rect.left + dx < 0:
            dx = -self.rect.left
        if self.rect.right + dx > screen_width:
            dx = screen_width - self.rect.right
        if self.rect.bottom + dy > screen_height - 110:
            self.vel_y = 0
            self.jump = False
            dy = screen_height - 110 - self.rect.bottom

        # ensure players face each other
        if target.rect.centerx > self.rect.centerx:
            self.flip = False
        else:
            self.flip = True

        # apply attack cooldown
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

        # update player position
        self.rect.x += dx
        self.rect.y += dy

    # handle animation updates
    def update(self):
        # check what action the player is performing
        if self.health <= 0:
            self.health = 0
            self.alive = False
            self.update_action(6)  # 6:death
        elif self.hit:
            self.update_action(5)  # 5:hit
        elif self.attacking:
            if self.attack_type == 1:
                self.update_action(3)  # 3:attack1
            elif self.attack_type == 2:
                self.update_action(4)  # 4:attack2
        elif self.jump:
            self.update_action(2)  # 2:jump
        elif self.running:
            self.update_action(1)  # 1:run
        else:
            self.update_action(0)  # 0:idle

        animation_cooldown = 50
        # update image
        self.image = self.animation_list[self.action][self.frame_index]
        # check if enough time has passed since the last update
        if pygame.time.get_ticks() - self.update_time > animation_cooldown:
            self.frame_index += 1
            self.update_time = pygame.time.get_ticks()
        # check if the animation has finished
        if self.frame_index >= len(self.animation_list[self.action]):
            # if the player is dead then end the animation
            if not self.alive:
                self.frame_index = len(self.animation_list[self.action]) - 1
            else:
                self.frame_index = 0
                # check if an attack was executed
                if self.action == 3 or self.action == 4:
                    self.attacking = False
                    self.attack_cooldown = 20
                # check if damage was taken
                if self.action == 5:
                    self.hit = False
                    # if the player was in the middle of an attack, then the attack is stopped
                    self.attacking = False
                    self.attack_cooldown = 20

    def attack(self, target):
        if self.attack_cooldown == 0:
            # execute attack
            self.attacking = True
            self.attack_sound.play()
            attacking_rect = pygame.Rect(self.rect.centerx - (2 * self.rect.width * self.flip), self.rect.y,
                                         2 * self.rect.width, self.rect.height)
            if attacking_rect.colliderect(target.rect):
                target.health -= 10
                target.hit = True

    def update_action(self, new_action):
        # check if the new action is different to the previous one
        if new_action != self.action:
            self.action = new_action
            # update the animation settings
            self.frame_index = 0
            self.update_time = pygame.time.get_ticks()

    def draw(self, surface):
        img = pygame.transform.flip(self.image, self.flip, False)
        surface.blit(img, (self.rect.x - (self.offset[0] * self.image_scale), self.rect.y - (self.offset[1] * self.image_scale)))
