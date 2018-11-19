---
title: RNN and LSTM 기초
date: 2018-11-17
categories: Study
tags: RNN LSTM
---
팀 내부 study용으로 간략히 정리한 내용입니다.

## RNN의 직관적 이해

- RNN의 은닉층은 DNN과 달리 이전 정보를 기억하는 역할
- 은닉층은 $$h_t$$나 $$s_t$$로 표기 (s의 의미는 state)
  - 네트워크의 메모리 부분으로서, 이전 시간 스텝의 hidden state 값과 $$t$$ 시점의 입력값에 의해 계산
- 입력층, 은닉층간의 weight는 $$U$$나 $$W_{xh}$$로 표기
- $$t-1$$시점 은닉층, $$t$$시점 은닉층 간의 weight는 $$W$$나 $$W_{hh}$$로 표기
- 은닉층, 출력층간의 weight는 $$V$$나 $$W_{hy}$$로 표기
- 가중치가 공유되므로, 파라매터 개수가 시간의 흐름과 무관하게 일정
- RNN의 은닉층은 DNN과 달리 이전 정보를 기억하는 역할
  - 각 layer마다의 파라미터 값들이 전부 다 다른 기존 신경망 구조와 달리, RNN은 모든 시간 스텝에 대한 가중치 $$U, V, W$$를 공유
  - 이는 RNN이 각 스텝마다 입력값만 다를 뿐 거의 똑같은 계산을 하고 있다는 것을 보여줌



![RNN]({{ "/assets/imgs/2018-11-17-RNN.png" | absolute_url }})

![RNN2]({{ "/assets/imgs/2018-11-17-RNN-unrolled.png" | absolute_url }})

## RNN의 문제점
- 예시
  - **나는** 그제 철수, 영희와 함께 경방 타임스퀘어 버스 정류장 앞에서 뛰면서 즐겁게 **놀았다.**
  - **나는 놀았다** 가 핵심
  - RNN은 이런 문장에 취약, why?  
- long-term dependency: $$x_1$$의 영향력이 점점 약해지면서 결국 소멸
- $$V$$는 출력에만 영향을 주지만 $$W, U$$는 은닉층에 영향을 끼침
- 타임스텝이 길수록 그레이던트 값이 기하급수적으로 작아져 0에 가까워짐. (gradient vanishing)
- 반대로 기하급수적으로 커지는 gradient exploding이 일어날 수도 있음

## LSTM의 직관적 이해
- 1997년 Hochreiter가 제안
- 가장 널리 쓰이는 버전은 2000년 Gers가 제안한 망각(forget) 게이트 추가, peephole 추가 버전
- 초기 버전 2개 게이트, 현 버전 3개 게이트 (입력 / 출력 / 망각)
- 핵심 아이디어
  - RNN은 hidden state를 다음 타입 스텝으로 그대로 보냄
  - LSTM은 입력 게이트, 출력 게이트와 메모리 불록의 역할을 하는 cell state를 추가하여 초기 시점에서의 데이터가 잘 전달될 수 있도록 함
  - 현재 cell state = 이전 cell state + 입력 게이트에서 걸러진 일부 입력단(hidden state)
    - $$c_t = c_{t-1} + g \circ i $$
  - 현재 cell state에 tanh activation을 적용 후 출력 게이트에서 한 번 더 거름
    - $$s_t  = \tanh(c_t) \circ o$$
- 망각 게이트 추가 시에는 이전 cell state 내용을 일부 망각
  -  $$c_t = c_{t-1} \circ f + g \circ i $$
- peephole 추가 시에는 cell state를 3개 게이트에 알려주는 역할을 함

![LSTM]({{ "/assets/imgs/2018-11-17-LSTM.png" | absolute_url }})

- [추천 사이트: colah의 블로그](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

$$
\begin{align}
i & = \sigma(x_tU^i + s_{t-1} W^i) \\
f & = \sigma(x_t U^f +s_{t-1} W^f) \\
o & = \sigma(x_t U^o + s_{t-1} W^o) \\
g & = \tanh(x_t U^g + s_{t-1}W^g) \\
c_t & = c_{t-1} \circ f + g \circ i \\
s_t & = \tanh(c_t) \circ o
\end{align}
$$

## GRU의 직관적 이해

- 출력 게이트와 cell state가 사라짐 (리셋 게이트 $$r$$과 업데이트 게이트 $$z$$로 대)
  - 리셋 게이트: 새로운 입력을 이전 메모리와 어떻게 합칠지를 정해줌
  - 업데이트 게이트: 이전 메모리를 얼만큼 기억할지 정해줌. 리셋 게이트 값을 전부 1로 정해주고 업데이트 게이트를 전부 0으로 정한다면, RNN과 동일
- [추천 사이트: aikorea.org 블로그](https://aikorea.org/blog/rnn-tutorial-4/)

$$
\begin{aligned}
z &= \sigma(x_tU^z + s_{t-1} W^z) \\
r &= \sigma(x_t U^r +s_{t-1} W^r) \\
h &= tanh(x_t U^h + (s_{t-1} \circ r) W^h) \\
s_t &= (1 - z) \circ h + z \circ s_{t-1}
\end{aligned}
$$
