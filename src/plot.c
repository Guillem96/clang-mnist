#include "plot.h"
#include <stdlib.h>
#include <stdint.h>

#include <SDL2/SDL.h> 
#include <SDL2/SDL_image.h> 
#include <SDL2/SDL_timer.h> 

static uint8_t g_sdl_initialized = 0;

/* Utility functions */
static SDL_Surface* grayscale_surface(tensor_t* t, SDL_Renderer* renderer);
static SDL_Surface* rgb_surface(tensor_t* t, SDL_Renderer* renderer);

void imshow(tensor_t* t, const char* title)
{
    SDL_Window* win;
    SDL_Renderer* renderer;
    SDL_Rect dest;
    SDL_Surface* surface;
    SDL_Texture* texture;
    SDL_Event event;
    uint8_t close = 0;

    if (!g_sdl_initialized && (SDL_Init(SDL_INIT_EVERYTHING) != 0)) { 
        printf("[ERROR] initializing SDL: %s\n", SDL_GetError()); 
        exit(1);
    }
    g_sdl_initialized = 1;

    if (t->n_dims != 2 && t->n_dims != 3)
    {
        printf("[ERROR] Tensor has %d dimensions. Expected a tensor of 2 or 3 dimensions", 
                t->n_dims);
        exit(1);
    }

    if (t->n_dims == 3 && t->shape[2] != 3 && t->shape[2] != 1)
    {
        printf("[ERROR] We only support RGB or BW images. Provided image has %d channels\n",
                t->shape[2]);
        exit(1);
    }

    win = SDL_CreateWindow(title, 
                SDL_WINDOWPOS_CENTERED,
                SDL_WINDOWPOS_CENTERED,
                480, 480, 0);

    if (win == NULL)
    {
        printf("Could not create window: %s\n", SDL_GetError());
        exit(1);
    }

    renderer = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    surface = t->n_dims == 2 ? grayscale_surface(t, renderer) : rgb_surface(t, renderer);
    texture = SDL_CreateTextureFromSurface(renderer, surface);
    dest.x = 0; dest.y = 0; dest.w = 480; dest.h = 480;

    while (!close)
    {
        while (SDL_PollEvent(&event))
        {
            switch (event.type)
            {
            case SDL_QUIT:
            case SDL_KEYDOWN:
                close = 1;
                break;
            }
        }

        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, &dest);
        SDL_RenderPresent(renderer);
        SDL_Delay(50);
    }

    SDL_FreeSurface(surface);
    SDL_DestroyRenderer(renderer); 
    SDL_DestroyTexture(texture);
    SDL_DestroyWindow(win);
}


SDL_Surface* grayscale_surface(tensor_t* t, SDL_Renderer* renderer)
{
    SDL_Surface* surface;
    SDL_Color colors[256];
    unsigned char* data = (unsigned char*)malloc(tensor_numel(t));

    for (int i = 0; i < tensor_numel(t); i++)
        data[i] = (unsigned char)t->values[i];

    surface = SDL_CreateRGBSurfaceFrom((void*)data, 
                                       t->shape[1], t->shape[0], 
                                       8, t->shape[1], 0, 0, 0, 0);

    if (surface == NULL) {
        printf("[ERROR] Creating surface failed: %s\n", SDL_GetError());
        exit(1);
    }

    for(int i = 0; i < 256; i++)
        colors[i].r = colors[i].g = colors[i].b = i;

    SDL_SetPaletteColors(surface->format->palette, colors, 0, 256);
    return surface;
}

SDL_Surface* rgb_surface(tensor_t* t, SDL_Renderer* renderer)
{
    SDL_Surface* surface;
    unsigned char* data = (unsigned char*)malloc(tensor_numel(t));

    for (int i = 0; i < tensor_numel(t); i++)
        data[i] = (unsigned char)t->values[i];

    surface = SDL_CreateRGBSurfaceFrom((void*)data, 
                                       t->shape[1], t->shape[0], 
                                       24, t->shape[1] * 3, 
                                       0x00ff0000, 
                                       0x0000ff00, 
                                       0x000000ff, 
                                       0);

    if (surface == NULL) {
        printf("[ERROR] Creating surface failed: %s\n", SDL_GetError());
        exit(1);
    }
    return surface;
}
