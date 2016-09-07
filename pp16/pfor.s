	.file	"pfor.c"
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC1:
	.string	"en c[%d]=%f\n"
	.section	.text.unlikely,"ax",@progbits
.LCOLDB2:
	.section	.text.startup,"ax",@progbits
.LHOTB2:
	.p2align 4,,15
	.globl	main
	.type	main, @function
main:
.LFB11:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset 3, -16
	xorl	%eax, %eax
	subq	$768, %rsp
	.cfi_def_cfa_offset 784
	.p2align 4,,10
	.p2align 3
.L2:
	pxor	%xmm0, %xmm0
	movl	$0x00000000, 512(%rsp,%rax,4)
	cvtsi2ss	%eax, %xmm0
	movss	%xmm0, 256(%rsp,%rax,4)
	movss	%xmm0, (%rsp,%rax,4)
	addq	$1, %rax
	cmpq	$64, %rax
	jne	.L2
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L3:
	movss	512(%rsp,%rax), %xmm0
	addss	(%rsp,%rax), %xmm0
	addss	256(%rsp,%rax), %xmm0
	movss	%xmm0, 512(%rsp,%rax)
	addq	$4, %rax
	cmpq	$256, %rax
	jne	.L3
	xorl	%ebx, %ebx
	.p2align 4,,10
	.p2align 3
.L4:
	pxor	%xmm0, %xmm0
	movl	%ebx, %esi
	movl	$.LC1, %edi
	movl	$1, %eax
	cvtss2sd	512(%rsp,%rbx,4), %xmm0
	addq	$1, %rbx
	call	printf
	cmpq	$64, %rbx
	jne	.L4
	addq	$768, %rsp
	.cfi_def_cfa_offset 16
	xorl	%eax, %eax
	popq	%rbx
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE11:
	.size	main, .-main
	.section	.text.unlikely
.LCOLDE2:
	.section	.text.startup
.LHOTE2:
	.ident	"GCC: (GNU) 5.2.0"
	.section	.note.GNU-stack,"",@progbits
