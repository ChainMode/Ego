# Hoja de Ruta

## Fase 1: Producto Mínimo Viable (MVP)
- [x] Soporte de fuente de datos de directorio local
- [x] Soporte de modelo base único (Qwen 3.5 0.8B)
- [x] Soporte de edición única (normal)
- [x] Compilación básica de corpus (formato de conversación)
- [x] Pipeline de entrenamiento QLoRA
- [x] Herramientas CLI (ego init / train / run)

## Fase 2: Soporte Multi-Versión
- [ ] Cuatro ediciones por defecto (lite/normal/plus/super)
- [ ] Soporte multi-base (Qwen, Gemma, Phi)
- [ ] Ediciones personalizadas
- [ ] Gestión de versiones y reversión

## Fase 3: Expansión de Fuentes de Datos
- [ ] Soporte de fuente de datos URL
- [ ] Fuentes de datos en almacenamiento en la nube (S3/OSS)
- [ ] Mecanismo de plugins para fuentes de datos
- [ ] Recolección incremental de datos y entrenamiento incremental

## Fase 4: Inteligencia
- [ ] Evaluación automática de alineación
- [ ] Auto-ajuste de parámetros de entrenamiento
- [ ] Entrenamiento automático programado (scheduler)
- [ ] Interfaz de gestión Web UI

## Fase 5: Integración con iFay
- [ ] Adaptador de formato de datos iFay
- [ ] Integración del protocolo de carga Fayger
- [ ] Mecanismo de distribución de paquetes Ego
