"""Pipeline build_temporal_windows.

Construye ventanas temporales deslizantes sobre los datasets preprocesados
(market_prepared, enriched_prepared) para el entrenamiento de redes neuronales
recurrentes (RNN/GRU/LSTM).

Cada ventana es una secuencia de `sequence_length` sesiones consecutivas
que se usa como input del modelo. El pipeline:
- Genera arrays NumPy agrupados por ticker.
- Detecta y descarta ventanas que cruzan gaps temporales excesivos.
- Produce indices de ventanas persistibles (Parquet).
- Genera resumenes por split.

Los window_stores completos (con arrays) se mantienen como MemoryDataset
para uso directo por los DataLoaders de PyTorch.
"""
