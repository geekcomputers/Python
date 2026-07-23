import random
import pygame
import sys

# Initialisation de pygame
pygame.init()

# Définir les couleurs
WHITE = (255, 255, 255)
PASTEL_PINK = (255, 182, 193)
PINK = (255, 105, 180)
LIGHT_PINK = (255, 182, 193)
GREY = (169, 169, 169)

# Définir les dimensions de la fenêtre
WIDTH = 600
HEIGHT = 600
FPS = 30
CARD_SIZE = 100

# Créer la fenêtre
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Memory Game : Les Préférences de Malak")

# Charger les polices
font = pygame.font.Font(None, 40)
font_small = pygame.font.Font(None, 30)

# Liste des questions et réponses (préférences)
questions = [
    {
        "question": "Quelle est sa couleur préférée ?",
        "réponse": "Rose",
        "image": "rose.jpg",
    },
    {
        "question": "Quel est son plat préféré ?",
        "réponse": "Pizza",
        "image": "pizza.jpg",
    },
    {
        "question": "Quel est son animal préféré ?",
        "réponse": "Chat",
        "image": "chat.jpg",
    },
    {
        "question": "Quel est son film préféré ?",
        "réponse": "La La Land",
        "image": "lalaland.jpg",
    },
]

# Créer les cartes avec des questions et réponses
cards = []
for q in questions:
    cards.append(q["réponse"])
    cards.append(q["réponse"])

# Mélanger les cartes
random.shuffle(cards)

# Créer un dictionnaire pour les positions des cartes
card_positions = [(x * CARD_SIZE, y * CARD_SIZE) for x in range(4) for y in range(4)]


# Fonction pour afficher le texte
def display_text(text, font, color, x, y):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))


# Fonction pour dessiner les cartes
def draw_cards():
    for idx, pos in enumerate(card_positions):
        x, y = pos
        if visible[idx]:
            pygame.draw.rect(screen, WHITE, pygame.Rect(x, y, CARD_SIZE, CARD_SIZE))
            display_text(cards[idx], font, PINK, x + 10, y + 30)
        else:
            pygame.draw.rect(
                screen, LIGHT_PINK, pygame.Rect(x, y, CARD_SIZE, CARD_SIZE)
            )
            pygame.draw.rect(screen, GREY, pygame.Rect(x, y, CARD_SIZE, CARD_SIZE), 5)


# Variables du jeu
visible = [False] * len(cards)
flipped_cards = []
score = 0

# Boucle principale du jeu
running = True
while running:
    screen.fill(PASTEL_PINK)
    draw_cards()
    display_text("Score: " + str(score), font_small, PINK, 20, 20)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            col = x // CARD_SIZE
            row = y // CARD_SIZE
            card_idx = row * 4 + col

            if not visible[card_idx]:
                visible[card_idx] = True
                flipped_cards.append(card_idx)

                if len(flipped_cards) == 2:
                    if cards[flipped_cards[0]] == cards[flipped_cards[1]]:
                        score += 1
                    else:
                        pygame.time.delay(1000)
                        visible[flipped_cards[0]] = visible[flipped_cards[1]] = False
                    flipped_cards.clear()

    if score == len(questions):
        display_text(
            "Félicitations ! Vous êtes officiellement le plus grand fan de Malak.",
            font,
            PINK,
            100,
            HEIGHT // 2,
        )
        display_text(
            "Mais… Pour accéder au prix ultime (photo ultra exclusive + certificat de starlette n°1),",
            font_small,
            PINK,
            30,
            HEIGHT // 2 + 40,
        )
        display_text(
            "veuillez envoyer 1000$ à Malak Inc.",
            font_small,
            PINK,
            150,
            HEIGHT // 2 + 70,
        )
        display_text(
            "(paiement accepté en chocolat, câlins ou virement bancaire immédiat)",
            font_small,
            PINK,
            100,
            HEIGHT // 2 + 100,
        )
        pygame.display.update()
        pygame.time.delay(3000)
        running = False

    pygame.display.update()
    pygame.time.Clock().tick(FPS)

# Quitter pygame
pygame.quit()
sys.exit()
