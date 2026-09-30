# v16 짝 학습 독립 증거 검토

독립 검토자가 고정 학습 자료를 읽기 전용으로 확인했다. **무결성 PASS**이며,
이는 새 보상의 성능 개선 판정이 아니다.

- 현재 파일이 고정 소스20/20개 SHA-256과 일치한다.
- 각 arm 250iteration, 4,096환경×32step = 32,768,000 transition;
  학습 로그의 iteration0–249, 누적 timestep과 일치한다.
- 최종 `model_249.pt`, 학습 텍스트 로그, TensorBoard event의 SHA가
  기록과 일치하고 모델/Adam tensor가 유한하다. 두 최종 checkpoint의
  `iter`는249다.
- 초기91D 상태·관측·RNG·정책 텐서가 같고, 저장 설정의 차이는 결과
  경로와 contact-slip 가중치0/1뿐이다. 처음 저장된 학습 전 정책은
  `v16_init/model_0.pt`와 별도 초기 감사의 해시로 증명한다.
- 네 발 모두 terrain mesh와 flat collision plane의 필터된 접촉이
  비영이며, 학습 접촉 진단의 무효 비율은0이다.

주의: 각 학습 실행 폴더의 `model_0.pt`는 RSL의 **첫 업데이트 이후**
파일이다. 이를 학습 전 초기 checkpoint라고 잘못 부르지 않는다.
이 검토는 훈련 무결성만 판단하며 보류 지형 성능은
[별도 원시 감사](independent_final_audit.md)를 따른다.
