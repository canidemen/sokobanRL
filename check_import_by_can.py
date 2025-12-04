import gym
import gym_sokoban
import pygame
import numpy as np

env = gym.make('Sokoban-v2')
obs = env.reset()

# Initialize pygame
pygame.init()
img = env.render(mode='rgb_array')
screen = pygame.display.set_mode((img.shape[1] * 3, img.shape[0] * 3))
pygame.display.set_caption('Sokoban')
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    img = env.render(mode='rgb_array')
    # Scale up and display
    surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
    surf = pygame.transform.scale(surf, (img.shape[1] * 3, img.shape[0] * 3))
    screen.blit(surf, (0, 0))
    pygame.display.flip()
    
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
    
    clock.tick(5)  # 5 FPS

pygame.quit()
env.close()