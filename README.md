---
Binary Convolution Test Code
---

------



- version_3_ver2 : 학습 방볍의 변화, 2Stage 학습

- version_3_ver4 : Learnable Bias 및 BN을 그룹당 적용

- version_3_ver6 : Learnable Bias 및 BN을 그룹당 적용 후 BN 한번 더 적용

- version_3_ver7 : Learnable Bias 및 BN을 그룹당 적용 전 BN 한번 더 적용

- version_3_ver8 : Learnable Bias 및 LN(Layer Nom)을 그룹당 적용 후 BN 한번 더 적용
  (Bias -> LN -> Local Activation ->  Global Activation -> BN )

- version_3_ver9 : Learnable Bias 제외 LN(Layer Nom)을 그룹당 적용 후 BN 한번 더 적용
  ( LN ->  local Activation  -> Global Activation -> BN)

- version_3_ver10: Learnable Bias 및 LN(Layer Nom)을 그룹당 적용 후 BN 한번 적용 제외
  (Bias -> LN ->  local Activation  -> Global Activation  )

- version_3_ver11: Learnable Bias 를 그룹당 적용 후 BN 한번 적용
  (Bias -> LN ->  local Activation  )
  
- version_3_ver12: Local Activation 제외하고
  (Bias -> LN ->  Global Activation -> BN )
  
- version_3_ver13: Activation 순서를 바꿔서
  (Bias -> Local Activation -> LN ->  Global Activation -> BN )
  
- version_3_ver14: Bias추가
  (Bias -> Local Activation -> LN -> Bias -> Global Activation -> BN )
