/*
 *  training.h
 *
 *  Arxiu reutilitzat de l'assignatura de Computació d'Altes Prestacions de l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona
 *  Created on: 31 gen. 2019
 *  Last modified: fall 24 (curs 24-25)
 *  Author: ecesar, asikora
 *  Modified: Blanca Llauradó, Christian Germer
 *
 *  Descripció:
 *  Capçaleres de les funcions per entrenar la xarxa neuronal.
 */

#ifndef TRAINING_H
#define TRAINING_H

#include "../layer/layer.h"

extern int num_layers;
extern layer* lay;
extern int* num_neurons;
extern char** input;
extern float* cost;
extern float full_cost;
extern float tcost;
extern float alpha;
extern int n;
extern int debug;
extern float** desired_outputs;

void feed_input(int i);
void forward_prop();
void compute_cost(int i);
void back_prop(int p);
void update_weights(void);

#endif