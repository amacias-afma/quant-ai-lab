# Guía técnica del artículo — para defenderlo

Documento interno. No es parte del manuscrito. Su propósito es que puedas responder cualquier
pregunta sobre la matemática, los tests y los puntos atacables, sin depender de recordar cómo
llegamos a cada cosa.

**Todos los números citados aquí fueron recomputados al escribir este documento**, no copiados
del borrador. Las verificaciones están al final (§8).

---

## 1. La tesis, en una línea — y sus límites

> Un resultado estadísticamente convincente puede ser producido por el **procedimiento de ajuste**
> en vez de por los datos, y ninguna cantidad adicional de testeo estadístico lo detecta.

**Lo que el paper SÍ afirma:**

1. En este estudio, cuatro hallazgos aparentemente sólidos se retiraron, cada uno por una razón
   distinta.
2. El cuarto —"el anclaje estabiliza el estimador"— era **mecánicamente garantizado** por la
   penalización, con independencia de si el ancla contenía información.
3. Ese mecanismo es demostrable analíticamente y reproducible sintéticamente con ground truth
   conocido.
4. Un control barato (objetivo permutado con escala emparejada) lo detecta; ningún test
   estadístico adicional lo hace.

**Lo que el paper NO afirma, y hay que decirlo si preguntan:**

- No afirma nada sobre modelado de VaR. Es un vehículo (§8 lo dice explícitamente).
- No afirma que estos cuatro modos de falla sean los más comunes. n = 1 estudio.
- No afirma que el control sea nuestro. Es un test de aleatorización restringida con linaje de
  90 años (§3.5, Apéndice D).
- No afirma prevalencia en la literatura. Cinco papers por conveniencia no son una muestra.

---

## 2. El objeto matemático

### 2.1 Pérdida pinball

Para un cuantil de nivel `α` (usamos α = 0.01 y α = 0.05; VaR al 99% y 95%), con retorno
realizado `y` y pronóstico `q`:

```
L_α(y, q) = (y − q) · (α − 1{y < q})
```

Expandida:

- si `y ≥ q` (sin breach): `L = α · (y − q) ≥ 0`
- si `y < q` (breach):     `L = (1 − α) · (q − y) ≥ 0`

Es no negativa y **estrictamente consistente** para el cuantil α: su esperanza se minimiza de
forma única en el cuantil verdadero (Koenker & Bassett 1978; Gneiting 2011). Esto es lo que
justifica usarla para *rankear* modelos.

> **Por qué importa y por qué es un hallazgo del survey.** Cuatro de los cinco papers ML-VaR
> examinados rankean con algo que **no** es estrictamente consistente: tasas de excedencia,
> pérdidas tipo Lopez, o el p-valor de un test de cobertura condicional. Un modelo puede tener
> cobertura impecable y estar lejos del cuantil verdadero, porque el test de cobertura solo mira
> la *secuencia de violaciones*. Nosotros rankeamos con pinball todo el tiempo — ese defecto
> específico no es nuestro.

**Si preguntan por qué pinball y no FZ0:** FZ0 (Fissler & Ziegel 2016) es para el par conjunto
(VaR, ES). Nosotros solo pronosticamos VaR, así que pinball es la elección correcta. FZ0 se cita
para el caso conjunto, no se usa.

### 2.2 El objetivo anclado

```
L(θ) = pinball(y − Xθ)  +  w · ‖θ − a‖²
```

- `θ` : parámetros del modelo
- `a` : el **ancla** (prior clásico de VaR: Normal rodante `μ − z·σ`, o cuantil histórico rodante)
- `w` : peso de la penalización, elegido en validación sobre una grilla que **incluye w = 0**

**El estimador anida su propio baseline.** Como la grilla contiene w = 0, la especificación
anclada *contiene* a la no anclada. Por construcción no puede ser peor en validación; solo puede
perder por **error de selección**. Esto es central para §3.2 y hay que tenerlo claro.

---

## 3. La derivación de contracción — el corazón del paper

Esta es la parte que más te van a preguntar. Hay que poder hacerla en una pizarra.

### 3.1 El álgebra completa

Descenso de gradiente con paso `η` (lr):

```
θ_{t+1} = θ_t − η · ∇L(θ_t)
        = θ_t − η · [ g(θ_t) + 2w(θ_t − a) ]
```

donde `g(θ)` es el (sub)gradiente del término pinball. El gradiente de la penalización es
`∇_θ [w‖θ − a‖²] = 2w(θ − a)`.

Tomamos **dos corridas que difieren solo en la inicialización**: `θ_t` y `θ'_t`. Definimos la
separación `Δ_t = θ_t − θ'_t`.

```
Δ_{t+1} = θ_{t+1} − θ'_{t+1}

        = (θ_t − θ'_t)
          − η [ g(θ_t) − g(θ'_t) ]
          − η · 2w [ (θ_t − a) − (θ'_t − a) ]
```

**Y acá está todo el argumento.** En el último corchete, `a` aparece dos veces con signo opuesto
y **se cancela exactamente**:

```
(θ_t − a) − (θ'_t − a) = θ_t − θ'_t = Δ_t
```

Por lo tanto:

```
Δ_{t+1} = (1 − 2ηw) · Δ_t  −  η [ g(θ_t) − g(θ'_t) ]
                              └──────── término de datos ────────┘
```

Si se **omite el término de datos**, la recursión es puramente multiplicativa:

```
‖Δ_T‖  ≈  (1 − 2ηw)^T · ‖Δ_0‖
```

**Esta expresión depende de `w`, `η` y `T`. No contiene `a`.** Contraer hacia el óptimo verdadero
y contraer hacia basura reducen la dispersión entre semillas **por el mismo factor**.

### 3.2 Lo que la derivación omite — ahora MEDIDO, no asumido

**El signo `≈` no es cosmético.** Omitimos `η[g(θ_t) − g(θ'_t)]`.

Durante mucho tiempo el paper afirmó que ese término era despreciable **sin medirlo** — la
práctica exacta que el artículo critica, cometida en su propia derivación central. Se midió
(`scripts/measure_contraction.py`, 400 pasos, η = 0.05, grilla de 10 pesos). **El resultado
corrigió dos cosas.**

#### Corrección 1: la fórmula cruda está mal por un factor ~14

| cantidad | resultado |
|---|---|
| contracción solo por el término omitido (en `w = 0`) | **14×** — la fórmula predice cero |
| ratio absoluto `observado/predicho` | mediana **0,070** |
| **ratio relativo a `w = 0`** | mediana **1,05**, rango **[0,80 ; 1,11]** |

La fila `w = 0` es la diagnóstica: ahí la predicción es idénticamente 1 (sin penalización, sin
contracción predicha), así que **toda** la contracción observada (a 0,0728 de la separación
inicial) es el término omitido. El descenso solo, sin ancla, ya contrae 14×.

**Pero mira la tercera fila.** El paper **nunca cita una dispersión absoluta**: cada número que
reporta es un *ratio contra el baseline sin anclar*. Y el término omitido aporta un factor
aproximadamente constante, que **se cancela en ese cociente**. Para la cantidad que el paper
efectivamente usa, la aproximación es buena al 5% mediano y 20% en el peor caso.

**Ésa es la respuesta cuando ataquen la aproximación:** la fórmula es mal predictor de la
dispersión absoluta y buen predictor de lo que reportamos.

#### Corrección 2: yo afirmé algo falso, y la medición lo desmintió

En la primera versión de esta guía escribí que las dos anclas debían dar trayectorias idénticas
*"a precisión numérica"*, porque `a` se cancela exactamente. **Eso era incorrecto.**

La cancelación es exacta **solo para el término de la penalización**. Pero el ancla sigue moviendo
cada iterado individualmente, así que cambia *dónde* está cada corrida, y por lo tanto cambia
`g(θ_a) − g(θ_b)` — el término omitido. **El ancla desaparece del término explícito y reentra por
la puerta de atrás, vía el término de datos.**

Medido:

| `w` | diferencia relativa máxima entre anclas |
|---|---|
| 0 | 0 (idénticas por construcción) |
| ≤ 0,017 | **≤ 3,4%** |
| 0,0308 | 13,7% |
| 0,0555 | 31,9% |
| 0,1 | **50,0%** |

**La afirmación correcta es: independiente del ancla a primer orden, con una dependencia de
segundo orden que crece con `w`.**

**¿Esto rompe el argumento? No, y los números dicen por qué:**

1. A `w = 0,1` las anclas difieren 50% mientras la contracción misma es 44–56×. El residuo es
   pequeño frente al efecto.
2. **El residuo corre en la dirección equivocada para rescatar el prior informativo**: el ancla
   basura contrae la separación de parámetros *un poco menos*, no más. Si el prior real tuviera
   ventaja informacional, esperaríamos lo contrario.
3. En el rango de pesos donde el estudio real operaba, la diferencia es ≤ 3,4%.

**Si te preguntan por esto, la respuesta corta es:** "medimos la aproximación en vez de asumirla;
resultó exacta a primer orden y con un residuo de segundo orden que crece con el peso, y ese
residuo no favorece al prior informativo."

### 3.3 La condición de presupuesto finito — sin ella el resultado es falso

El objetivo es **convexo**. Entonces con `T → ∞` toda inicialización converge al mismo óptimo y
la dispersión entre semillas va a cero **para cualquier `w`, incluido `w = 0`**.

La fórmula describe el **régimen pre-convergencia**: presupuesto finito de pasos, o
equivalentemente early stopping.

**Eso no es un artificio de la demostración.** Es el régimen en que el estudio real operaba,
porque su regla de parada cortaba el entrenamiento antes de converger (§5, defecto 1). La
demostración sintética reproduce esa condición en vez de asumirla ausente.

> Si un revisor dice *"con más pasos esto desaparece"*: tiene razón, y el paper lo dice. Ese es
> exactamente el punto — la estabilidad observada era un artefacto del presupuesto de
> entrenamiento, no una propiedad del ancla.

### 3.4 Verificación numérica de la fórmula

Con `η = 0.05`, `T = 400` (los valores reales de la demo):

| `w` | factor `(1−2ηw)^T` | ratio predicho `1/factor` | verdad observada | basura observada |
|---|---|---|---|---|
| 0.0308 | 0.29115 | **3.43** | 2.39× | 6.09× |
| 0.0555 | 0.10794 | **9.26** | 5.46× | 17.81× |
| 0.1000 | 0.01795 | **55.71** | 50.40× | 79.23× |

La curva analítica queda **entre** las dos series observadas. Eso es lo esperado: predice el orden
de magnitud y la forma, no el valor exacto, porque omitimos el término de datos. Es la línea
punteada del panel (b) de la Figura 2.

**El argumento visual:** una curva derivada solo del gradiente de la penalización, que no puede
ver hacia dónde se contrajo nada, predice ambas series.

### 3.5 Un detalle de diseño que te pueden preguntar

El prior barajado usa una **semilla fija (20260819)**, de modo que el objetivo permutado es
**idéntico para todas las semillas del modelo**.

¿Por qué importa? Porque si el barajado cambiara por semilla, `a` dejaría de ser un punto fijo y
la cancelación de §3.1 no aplicaría: cada corrida se contraería hacia un punto distinto y no
habría reducción de dispersión. **El control está construido para preservar exactamente la
propiedad que hace funcionar el mecanismo, y destruir solo la información.** Eso es lo que
"aleatorización restringida" significa.

---

## 4. Los cuatro hallazgos y por qué cayó cada uno

### 4.1 "El anclaje mejora la pérdida fuera de muestra" — multiplicidad

**Evidencia inicial:** 3 de 26 comparaciones rechazan al 5% sin corregir.

**El test:** Diebold–Mariano. Diferencial de pérdida `d_t = L_a(t) − L_b(t)`, se testea `E[d] = 0`.

```
DM = d̄ / sqrt( LRV / n )
```

donde LRV es la varianza de largo plazo HAC con núcleo Bartlett (Newey–West):

```
LRV = γ_0 + 2 · Σ_{k=1}^{L} (1 − k/(L+1)) · γ_k        (L = 5)
```

Se aplica la corrección de muestra pequeña de Harvey–Leybourne–Newbold.

**Por qué cayó:** corrección de Holm. Se ordenan los p-valores ascendentes y se rechaza `p_(i)`
solo si `p_(i) ≤ α/(m−i+1)`, secuencialmente. Controla FWER. **Sobrevive 1 de 26.**

**Y peor:** para cuando reportamos esto, el bloque de test se había puntuado cuatro veces, así que
ninguna comparación conserva estatus fuera de muestra.

> **Ataque posible: "¿por qué la familia es 26 y no 1.959?"** Respuesta: porque 26 es la elección
> **más favorable a nuestra propia hipótesis** — una familia más grande empeora la corrección. Es
> decir, elegimos el conteo que nos beneficia y aun así el hallazgo cayó.

**Un detalle técnico que costó 8 celdas:** cuando dos modelos producen pronósticos idénticos,
`d ≡ 0`, la LRV es 0 y el estadístico DM explota. Eso ocurría exactamente en las celdas donde
validación apagaba el ancla (`w = 0`) — el resultado más informativo disponible. Ahora se devuelve
`(0, 0.5)` en ese caso degenerado.

### 4.2 "La selección de peso es indistinguible del azar" — potencia

**Evidencia inicial:** de 16 comparaciones donde validación eligió `w > 0`, **6 (37,5%)** fueron
peores que `w = 0`. Binomial exacto dos colas `p = 0,4545`; IC 95% exacto `[0,152 ; 0,646]`.

**Por qué NO se puede concluir "es ruido":** a n = 16 el diseño solo detecta proporciones
`≤ 0,147` o `≥ 0,853` con 80% de potencia. La potencia contra el efecto observado es **9,5%**.
Para 80% harían falta **125 comparaciones**.

Traducido: *un procedimiento de selección con un error genuinamente útil del 30% habría pasado
desapercibido el 91% de las veces.*

**La afirmación correcta es "indeterminado", no "azar".**

**Y hay una capa más, que es la que un revisor de ML va a levantar** — Cawley & Talbot (2010):

> *"the effects of this form of over-fitting are often of **comparable magnitude to differences in
> performance between learning algorithms**"*

Nuestros edges observados son 1–8% del nivel de pérdida. Si el sobreajuste de selección opera a
esa misma escala, **la comparación se hizo dentro del piso de ruido de su propio paso de
selección**. Eso es más específico que "subpotenciado".

Ellos también notan que la superficie del criterio suele ser un *"broad valley"*: un
hiperparámetro mal elegido puede generalizar bien igual. Mejor explicación de nuestro 37,5% que la
que dimos primero.

### 4.3 "Más capacidad empeora la precisión" — el propio ruido del modelo

Un MLP de 8.641 parámetros perdió contra un modelo lineal de 4 parámetros en 4 de 5 activos.

**Por qué cayó, por tres vías independientes:**

1. **0 de 5** brechas exceden el IQR entre semillas combinado de los dos modelos comparados.
2. El MLP heredó learning rate y presupuesto de épocas del modelo lineal y **nunca fue afinado**;
   la convergencia no se registró, así que la afirmación es infalsable.
3. A n = 5 el diseño **no puede detectar ningún efecto** al 80% de potencia. Binomial 4/5 da
   `p = 0,375`.

Retirado, no resuelto. Es distinto.

### 4.4 "El anclaje estabiliza el estimador" — tautología

**Este era el resultado más fuerte del proyecto.** IQR entre semillas menor en 19/20, 21/23, 25/27
y **15/16** comparaciones. Conjunto primario: sign test **p = 5,19e-04**, ratio mediano **13,5×**
(IC bootstrap 95% 4,6–20,7, n = 16). Y una dosis-respuesta limpia entre el peso seleccionado y el
efecto: **Spearman ρ = +0,585, p = 4,1e-09, n = 85**.

**Por qué cayó:** §3 de este documento. La penalización L2 contrae toda semilla hacia el **mismo
punto fijo**, sea cual sea ese punto. **La dosis-respuesta que tomamos como corroboración es la
firma del artefacto.**

**El control:** reajustar con el prior real **permutado en el tiempo**. Idéntica media, desviación
y distribución marginal; correlación con el retorno absoluto de mañana 0,0007 contra 0,041.
Emparejado en magnitud, despojado de información.

El prior no informativo iguala o supera al real en 3 de 6 celdas; **Wilcoxon p = 0,844**.
Bootstrapeado sobre comparaciones: real **1,9× (IC 1,0–7,8)** contra barajado **2,2× (IC
0,9–9,9)** — se solapan casi por completo.

**La trampa que casi nos salva el hallazgo:** agrupar *todos* los controles da 1,32× contra 2,36×
para priors informativos, que parece una brecha real. La produce enteramente un control que se
contrae hacia un objetivo **fuera de escala**, que pelea contra los datos en vez de contraerse
dentro de ellos. Solo la comparación **emparejada en escala** es diagnóstica, y es nula.

Ojala & Garriga (2010) ya enunciaban el principio: *"which properties of the original data are
preserved in the randomization test"* determina la distribución nula. Lo descubrimos tropezando;
tiene nombre desde 2010.

---

## 5. El aparato estadístico — qué prueba y qué NO prueba cada test

| test | H₀ | qué detecta | qué NO detecta |
|---|---|---|---|
| **Kupiec POF** | tasa de breach = α | frecuencia incorrecta | agrupamiento temporal |
| **Christoffersen ind.** | `π₀₁ = π₁₁` (cadena de Markov) | breaches agrupados | tasa incorrecta |
| **Christoffersen CC** | ambas (χ², df = 2) | el gate estándar | distancia al cuantil verdadero |
| **Diebold–Mariano** | `E[d] = 0` | diferencia de pérdida | multiplicidad |
| **Holm** | — | controla FWER | efectos pequeños (pierde potencia) |
| **MCS (HLN)** | igual habilidad predictiva | qué modelos sobreviven | **nada, si n es chico** |
| **Sign / binomial exacto** | `p = 0.5` | dirección consistente | magnitud |
| **Spearman ρ** | sin asociación monótona | dosis-respuesta | **causalidad ni mecanismo** |
| **Wilcoxon pareado** | mediana de diferencias = 0 | diferencia pareada | **casi nada a n = 6** |
| **Bootstrap percentil** | — | incertidumbre de la mediana | se satura al rango observado si n es chico |

**Los dos que hay que saber explicar en detalle:**

**MCS (Hansen–Lunde–Nason 2011):** elimina modelos iterativamente hasta que no se rechaza la
hipótesis de igual habilidad predictiva; la distribución se obtiene por bootstrap de bloques
móviles. A n pequeño retiene **todo**. En nuestro caso retuvo los nueve modelos, incluido uno
demostrablemente mal calibrado. La lectura correcta es la de sus propios autores: *"uninformative
data yield a MCS with many models"* — no es que los modelos sean equivalentes, es que los datos no
alcanzan para separarlos.

**Bootstrap percentil sobre comparaciones:** la unidad de remuestreo es la **comparación**, no la
semilla, porque la cantidad reportada ("ratio mediano 13,5×") es una mediana *a través de
comparaciones*. Un intervalo a nivel de semilla respondería otra pregunta y requeriría las
pérdidas por semilla, que el pipeline original no persistió. **Los intervalos son anchos, y eso es
parte del hallazgo.**

---

## 6. Los puntos debatibles — ordenados por peligrosidad

### 🔴 A. El test empírico del prior barajado tiene n = 6

**El ataque más fuerte que existe contra el paper**, y hay que tenerlo preparado.

*"§3.2 les prohíbe concluir de un nulo a n = 16 con 9,5% de potencia. Pero en §3.4 concluyen
'indistinguibles' de un Wilcoxon con p = 0,844 a n = 6, que tiene aún menos potencia. ¿No es el
mismo error?"*

**Respuesta honesta, en este orden:**

1. **Sí, si el test empírico fuera la única evidencia, sería el mismo error.** Un no-rechazo a
   n = 6 no prueba nada.
2. **Por eso existe §3.5.** La afirmación no descansa en el Wilcoxon. Descansa en (i) la
   derivación analítica, donde `a` se cancela exactamente, y (ii) la demostración sintética con
   ground truth conocido, 10 pesos × 40 semillas, sign test pareado p = 0,021.
3. El test empírico a n = 6 es **corroboración, no evidencia**. §8 lo dice: *"decisive about
   mechanism, not about magnitude."*
4. La asimetría con §3.2 es real y defendible: en §3.2 afirmábamos un **nulo empírico** sin
   mecanismo; en §3.4 afirmamos un **mecanismo derivado** que además se observa. No es el mismo
   tipo de afirmación.

**Si insisten:** conceder que la sección debería reordenarse para que la derivación venga *antes*
que el Wilcoxon, no después. Es una crítica de presentación válida.

### 🟠 B. La derivación omite un término

Ver §3.2 de este documento, que ahora trae la **medición**, no la aserción.

**Respuesta corta:** el término omitido aporta una contracción de 14× por sí solo, así que la
fórmula cruda subpredice por ese factor. Pero el paper solo cita *ratios contra el baseline sin
anclar*, y ahí el término omitido se cancela: la predicción es exacta al 5% mediano.

**Y una concesión que hay que dar sin que la pidan:** la cancelación del ancla es exacta solo para
el término de penalización. El ancla reentra por el término de datos, con una dependencia de
segundo orden que llega al 50% en el peso más grande. Lo medimos, está en el paper, y el residuo
no favorece al prior informativo.

### 🟠 C. "No hay test set, entonces nada es fuera de muestra"

*"Ustedes mismos dicen que el proyecto tiene varios bloques de validación y ningún test. ¿Cómo
pueden afirmar algo?"*

**Respuesta:** la contaminación destruye las afirmaciones de **pronóstico**, que están todas
retiradas. La afirmación que sobrevive —§3.4/§3.5— no es una afirmación de pronóstico fuera de
muestra. Es una afirmación sobre un **mecanismo**, verificada analíticamente y sintéticamente con
datos simulados que no tienen nada que ver con el bloque contaminado. **La demostración sintética
no toca datos de mercado en absoluto.**

### 🟠 D. "La demostración sintética está armada para dar ese resultado"

**Respuesta:** la derivación analítica predice el resultado **antes** de cualquier simulación. La
demo confirma el álgebra, no la descubre. Y crucialmente, la demo **distingue los ejes en vez de
colapsarlos**: el ancla verdadera *mejora* la pérdida fuera de muestra mientras la basura la
*empeora*. Si estuviera armada, ambas se verían iguales en todo.

### 🟡 E. El IQR entre semillas es un estimador crudo

A 10 semillas, el IQR es ruidoso, y un **ratio** de IQRs lo es aún más (cociente de cantidades
ruidosas, distribución de cola pesada). Por eso el IC bootstrap de 13,5× va de 4,6 a 20,7 — muy
ancho. **El paper lo dice y lo cuenta como parte del hallazgo.** La demo sintética usa 40 semillas
por celda precisamente para reducir esto.

### 🟡 F. Warm start entre reajustes crea dependencia

El protocolo reajusta mensualmente sobre ventana expansiva con **warm start** desde los pesos del
período anterior. Eso induce dependencia serial entre reajustes dentro de una misma semilla.
**No lo modelamos.** Es una limitación honesta que no está en §8 y probablemente debería estar.

### 🟡 G. Partición cronológica única

Cawley & Talbot recomiendan múltiples particiones. Barajar está prohibido (serie temporal), pero
**múltiples orígenes cronológicos sí estaban disponibles y no los corrimos**. Está en §8 como
limitación, no como defensa.

### 🟢 H. "El benchmark GARCH tenía un bug a su favor"

Al revés: el bug lo **perjudicaba**. El cuantil t estandarizado dividía en vez de multiplicar por
`sqrt((ν−2)/ν)`, inflando el VaR por un factor `ν/(ν−2)`. O sea, **el baseline que estábamos
ganando venía con handicap**. Corregido, con tests. Está en §1.1 como parte de la procedencia.

### 🟢 I. "n = 3 defectos en la misma dirección es sospechoso"

`p = 0,125` una cola bajo signos independientes. **Por nuestro propio estándar de §3.2, n = 3 no
sostiene una conclusión, y no la sacamos.** Lo ofrecemos como mecanismo a testear con muestra
mayor. Que el paper se aplique su propio estándar en su contra es probablemente el pasaje que más
crédito le da.

---

## 7. Las preguntas que te van a hacer

**"¿Por qué debería importarme un resultado nulo?"**
No es el nulo lo que importa. Es que el hallazgo que sobrevivió a todo el testeo estadístico
—preregistro, 10 semillas, pérdida consistente, DM, MCS, gate de cobertura, replicación en cuatro
corridas, dosis-respuesta a p = 5,2e-04— cayó ante una pregunta que no es estadística: *¿el
mecanismo garantiza este resultado?*

**"¿Qué tiene de nuevo? El shrinkage es de manual."**
Nada del mecanismo es nuevo — es Stein/ridge, y lo citamos. Lo que documentamos es que el
artefacto es **reportable como hallazgo empírico**, que sobrevive un protocolo preregistrado, y
que un control de minutos lo separa. No reclamamos el control tampoco: es aleatorización
restringida, con linaje Fisher (1935) → Pitman (1937) → Ojala & Garriga (2010).

**"¿Entonces qué es lo suyo?"**
No encontramos una aplicación de ese control a una afirmación de **estabilidad** (en vez de
precisión o efecto de tratamiento). Lo escribimos como **nulo de búsqueda, no como hueco** — no
haberlo encontrado no prueba que no exista.

**"¿Esto aplica fuera de VaR?"**
Sí, y esa es la razón de §3.5. Aplica a cualquier estimador regularizado donde se ofrezca
estabilidad o reducción de varianza como evidencia de calidad. El mecanismo no sabe de finanzas.

**"¿Por qué debería creerles los números?"**
Ninguno está tipeado a mano. Salen de `results/*.csv`, y un comando falla si el archivo de
resúmenes se desincroniza de los CSV. Encontramos **dos archivos rancios** haciendo justamente esa
verificación — está en el Apéndice B, incluido el que contenía el par de valores retirados.

**"Encontraron cinco errores de cita en veinte. ¿Cómo confío en el resto?"**
Porque los encontramos y los reportamos. Las 47 referencias están verificadas contra el registro
publicado, ninguna afirmada de memoria. Uno de los errores tenía el argumento del paper citado
**al revés** — habría entrado como aliado fabricado.

**"¿Cuál es la acción concreta para alguien que despliega un modelo el lunes?"**
Antes de comparar pérdidas, reajustar contra un prior permutado en el tiempo —misma media, misma
varianza, misma marginal, cero información— y ver qué le pasa al beneficio reportado. Lo que
sobreviva es real; lo que no, era la penalización.

---

## 8. Verificación de los números citados

Recomputados al escribir este documento:

```
15/16 dos colas  : 0.000518798828125   (paper 5.2e-04)   ✓
 6/16 dos colas  : 0.454498291015625   (paper 0.454)     ✓
 9/10 dos colas  : 0.021484375         (paper 0.021)     ✓
  4/5 dos colas  : 0.375               (paper 0.375)     ✓
  3/3 una cola   : 0.125               (paper 0.125)     ✓
IC 95% de 6/16   : [0.1520, 0.6457]    (paper 0.152-0.646) ✓

Contracción analítica (η=0.05, T=400):
  w=0.0308  factor=0.29115  ratio=3.43
  w=0.0555  factor=0.10794  ratio=9.26
  w=0.1000  factor=0.01795  ratio=55.71
```

Enteros de divulgación (del ledger, no contados a mano): **1.959 evaluaciones sobre test, 16
celdas, 4 pasadas máximo, 0 celdas puntuadas una sola vez.**

Comandos para verificar antes de cualquier presentación:

```bash
python -m pytest -q                                  # 112 passed, 5 skipped
python -m value_at_risk.evaluation.ledger --summary  # los enteros de §4
python scripts/refresh_paper_figures.py --check      # debe salir con código 0
```

---

## 9. Lo que yo revisaría antes de defenderlo

Tres cosas que no están del todo cerradas y que un revisor cuidadoso puede levantar:

1. **El orden de §3.4.** El Wilcoxon a n = 6 aparece antes que la derivación analítica. Sería más
   sólido presentar el mecanismo primero y el empírico como corroboración. Es reordenar, no
   reescribir.
2. **La dependencia por warm start** no está en §8 y probablemente debería.
3. ~~La afirmación de que el término omitido es despreciable no está cuantificada.~~
   **HECHO.** Se midió (§3.2). Resultado: la fórmula cruda está mal por ~14×, la predicción del
   ratio relativo —lo único que el paper reporta— es exacta al 5% mediano, y **la medición
   falsificó una afirmación mía sobre invariancia respecto del ancla**. Ambas correcciones están
   en el paper y en §3.2 de esta guía. `scripts/measure_contraction.py`, tests en
   `tests/test_shrinkage_demo.py`.

   > Vale la pena registrar cómo salió esto: la afirmación falsa la escribí **yo, en la guía
   > preparada para defender el paper**, y sobrevivió hasta que alguien corrió el experimento.
   > Es la cuarta vez en este proyecto que una verificación desmiente algo que se daba por
   > obvio — y la primera en que el error estaba en el documento cuyo propósito era detectar
   > errores.
