' ===================================================================
'
'          Standalone Llama32 VB.NET Inference Engine
'
' ===================================================================

Imports System
Imports System.Collections.Generic
Imports System.Diagnostics
Imports System.IO
Imports System.Linq
Imports System.Text
Imports System.Runtime.InteropServices
Imports System.Text.RegularExpressions ' <-- Добавлен импорт для Regex

' === Класс-запускатор для надежной точки входа ===
Public Class Program
    Public Shared Sub Main(args As String())
        Llama32Standalone.RunApplication(args)
    End Sub
End Class


Public Module Llama32Standalone

    ' --- Type Definitions ---
    Public Enum DType
        Float16
        Float32
    End Enum

    ' --- Configuration ---
    Public NotInheritable Class Llama32Config
        Private Sub New()
        End Sub
        Public Const NumLayers As Integer = 16
        Public Const NVocab As Integer = 128256
        Public Const NLayers As Integer = 16
        Public Const DEmbd As Integer = 2048
        Public Const NHeads As Integer = 32
        Public Const NKvHeads As Integer = 8
        Public Const DHead As Integer = 64
        Public Const DMlp As Integer = 8192
        Public Const RmsNormEps As Single = 1.0e-05F
    End Class

    ' --- Utility Assertion ---
    Private Sub Llama32Assert(condition As Boolean, Optional message As String = "Assertion failed")
        If Not condition Then
            Dim st = New System.Diagnostics.StackTrace(True)
            Dim frame = st.GetFrame(1)
            Dim line = frame.GetFileLineNumber()
            Dim file = frame.GetFileName()
            Console.Error.WriteLine(vbLf & $"LLAMA32_ASSERT: {file}:{line}: {message}")
            Environment.Exit(1)
        End If
    End Sub

    ' --- FP16 Conversion Functions ---
    Public NotInheritable Class Float16Converter
        Private Sub New()
        End Sub
        Private Shared ReadOnly Fp16ToFp32Table As Single()

        Shared Sub New()
            Fp16ToFp32Table = New Single(65535) {}
            For i As Integer = 0 To 65535
                Fp16ToFp32Table(i) = ConvertFp16ToFp32(CUShort(i))
            Next
        End Sub

        Public Shared Function Fp16ToFp32(h As UShort) As Single
            Return Fp16ToFp32Table(h)
        End Function

        Private Shared Function Fp32FromBits(w As UInteger) As Single
            ' С отключенной проверкой на переполнение CInt будет работать как простая переинтерпретация бит
            Return BitConverter.Int32BitsToSingle(CInt(w))
        End Function

        Private Shared Function Fp32ToBits(f As Single) As UInteger
            ' С отключенной проверкой на переполнение CInt будет работать как простая переинтерпретация бит
            Return CUInt(BitConverter.SingleToInt32Bits(f))
        End Function

        Private Shared Function ConvertFp16ToFp32(h As UShort) As Single
            Dim w As UInteger = CUInt(h) << 16
            Dim sign As UInteger = w And &H80000000UI
            ' С отключенной проверкой на переполнение сложение будет "заворачиваться"
            Dim two_w As UInteger = w + w

            Dim exp_offset As UInteger = CUInt(&HE0) << 23
            Dim exp_scale As Single = Fp32FromBits(&H7800000UI)
            Dim normalized_value As Single = Fp32FromBits((two_w >> 4) + exp_offset) * exp_scale

            Dim magic_mask As UInteger = CUInt(126) << 23
            Dim magic_bias As Single = 0.5F
            Dim denormalized_value As Single = Fp32FromBits((two_w >> 17) Or magic_mask) - magic_bias

            Dim denormalized_cutoff As UInteger = CUInt(1) << 27
            Dim denormalized_bits = Fp32ToBits(denormalized_value)
            Dim normalized_bits = Fp32ToBits(normalized_value)
            Dim result_bits As UInteger = sign Or If(two_w < denormalized_cutoff, denormalized_bits, normalized_bits)
            Return Fp32FromBits(result_bits)
        End Function

        Public Shared Function Fp32ToFp16(f As Single) As UShort
            Dim scale_to_inf As Single = Fp32FromBits(&H77800000UI)
            Dim scale_to_zero As Single = Fp32FromBits(&H8800000UI)
            Dim base_val As Single = (Math.Abs(f) * scale_to_inf) * scale_to_zero

            Dim w As UInteger = Fp32ToBits(f)
            ' С отключенной проверкой на переполнение сложение будет "заворачиваться"
            Dim shl1_w As UInteger = w + w
            Dim sign As UInteger = w And &H80000000UI
            Dim bias As UInteger = shl1_w And &HFF000000UI
            If bias < &H71000000UI Then
                bias = &H71000000UI
            End If

            base_val = Fp32FromBits((bias >> 1) + &H7800000UI) + base_val
            Dim bits As UInteger = Fp32ToBits(base_val)
            Dim exp_bits As UInteger = (bits >> 13) And &H7C00UI
            Dim mantissa_bits As UInteger = bits And &HFFFUI
            Dim nonsign As UInteger = exp_bits + mantissa_bits
            Return CUShort((sign >> 16) Or If(shl1_w > &HFF000000UI, &H7E00UI, nonsign))
        End Function
    End Class

    ' --- Neural Network Operations ---
    Public NotInheritable Class Ops
        Private Sub New()
        End Sub
        Public Shared Sub Embed(tokens As Integer(), embTable As Array, output As Array, nCtx As Integer, dEmbd As Integer, dtype As DType)
            Dim elementSize As Integer = If(dtype = DType.Float16, Marshal.SizeOf(GetType(UShort)), Marshal.SizeOf(GetType(Single)))
            For i As Integer = 0 To nCtx - 1
                Dim embTableIdx As Integer = tokens(i)
                Dim srcOffset As Integer = embTableIdx * dEmbd
                Dim destOffset As Integer = i * dEmbd
                Buffer.BlockCopy(embTable, srcOffset * elementSize, output, destOffset * elementSize, dEmbd * elementSize)
            Next
        End Sub

        Public Shared Sub RmsNorm(inp As Array, weight As Array, output As Array, nCtx As Integer, dEmbd As Integer, eps As Single, dtype As DType)
            If dtype = DType.Float16 Then
                RmsNormF16(CType(inp, UShort()), CType(weight, UShort()), CType(output, UShort()), nCtx, dEmbd, eps)
            Else
                RmsNormF32(CType(inp, Single()), CType(weight, Single()), CType(output, Single()), nCtx, dEmbd, eps)
            End If
        End Sub

        Private Shared Sub RmsNormF32(inp As Single(), weight As Single(), output As Single(), nCtx As Integer, dEmbd As Integer, eps As Single)
            For i As Integer = 0 To nCtx - 1
                Dim sumSquares As Single = 0.0F
                Dim offset As Integer = i * dEmbd
                For j As Integer = 0 To dEmbd - 1
                    sumSquares += inp(offset + j) * inp(offset + j)
                Next
                Dim rms As Single = CSng(Math.Sqrt(sumSquares / dEmbd))
                Dim rsqrt As Single = 1.0F / (rms + eps)
                For j As Integer = 0 To dEmbd - 1
                    output(offset + j) = inp(offset + j) * rsqrt * weight(j)
                Next
            Next
        End Sub

        Private Shared Sub RmsNormF16(inp As UShort(), weight As UShort(), output As UShort(), nCtx As Integer, dEmbd As Integer, eps As Single)
            For i As Integer = 0 To nCtx - 1
                Dim sumSquares As Single = 0.0F
                Dim offset As Integer = i * dEmbd
                For j As Integer = 0 To dEmbd - 1
                    Dim val As Single = Float16Converter.Fp16ToFp32(inp(offset + j))
                    sumSquares += val * val
                Next
                Dim rms As Single = CSng(Math.Sqrt(sumSquares / dEmbd))
                Dim rsqrt As Single = 1.0F / (rms + eps)
                For j As Integer = 0 To dEmbd - 1
                    Dim valIn As Single = Float16Converter.Fp16ToFp32(inp(offset + j))
                    Dim valW As Single = Float16Converter.Fp16ToFp32(weight(j))
                    output(offset + j) = Float16Converter.Fp32ToFp16(valIn * rsqrt * valW)
                Next
            Next
        End Sub

        Public Shared Sub Residual(inp0 As Array, inp1 As Array, output As Array, nCtx As Integer, dEmbd As Integer, dtype As DType)
            Dim count As Integer = nCtx * dEmbd
            If dtype = DType.Float16 Then
                Dim i0 = CType(inp0, UShort()), i1 = CType(inp1, UShort()), o = CType(output, UShort())
                For i As Integer = 0 To count - 1
                    o(i) = Float16Converter.Fp32ToFp16(Float16Converter.Fp16ToFp32(i0(i)) + Float16Converter.Fp16ToFp32(i1(i)))
                Next
            Else
                Dim i0 = CType(inp0, Single()), i1 = CType(inp1, Single()), o = CType(output, Single())
                For i As Integer = 0 To count - 1
                    o(i) = i0(i) + i1(i)
                Next
            End If
        End Sub

        Public Shared Sub MulInplace(inp0 As Array, inp1 As Array, nCtx As Integer, dEmbd As Integer, startPos As Integer, dtype As DType)
            If dtype = DType.Float16 Then
                Dim i0 = CType(inp0, UShort()), i1 = CType(inp1, UShort())
                For i As Integer = startPos To nCtx - 1
                    Dim offset As Integer = i * dEmbd
                    For j As Integer = 0 To dEmbd - 1
                        i0(offset + j) = Float16Converter.Fp32ToFp16(Float16Converter.Fp16ToFp32(i0(offset + j)) * Float16Converter.Fp16ToFp32(i1(offset + j)))
                    Next
                Next
            Else
                Dim i0 = CType(inp0, Single()), i1 = CType(inp1, Single())
                For i As Integer = startPos To nCtx - 1
                    Dim offset As Integer = i * dEmbd
                    For j As Integer = 0 To dEmbd - 1
                        i0(offset + j) *= i1(offset + j)
                    Next
                Next
            End If
        End Sub

        Public Shared Sub LmHeadProj(inp As Array, weight As Array, output As Single(), nVocab As Integer, nCtx As Integer, dEmbd As Integer, dtype As DType)
            If dtype = DType.Float16 Then
                LmHeadProjF16(CType(inp, UShort()), CType(weight, UShort()), output, nVocab, nCtx, dEmbd)
            Else
                LmHeadProjF32(CType(inp, Single()), CType(weight, Single()), output, nVocab, nCtx, dEmbd)
            End If
        End Sub

        Private Shared Sub LmHeadProjF32(inp As Single(), weight As Single(), output As Single(), nVocab As Integer, nCtx As Integer, dEmbd As Integer)
            Dim i As Integer = nCtx - 1
            Dim inpOffset As Integer = i * dEmbd
            For j As Integer = 0 To nVocab - 1
                Dim dotProd As Single = 0.0F
                Dim weightOffset As Integer = j * dEmbd
                For k As Integer = 0 To dEmbd - 1
                    dotProd += inp(inpOffset + k) * weight(weightOffset + k)
                Next
                output(j) = dotProd
            Next
        End Sub

        Private Shared Sub LmHeadProjF16(inp As UShort(), weight As UShort(), output As Single(), nVocab As Integer, nCtx As Integer, dEmbd As Integer)
            Dim i As Integer = nCtx - 1
            Dim inpOffset As Integer = i * dEmbd
            For j As Integer = 0 To nVocab - 1
                Dim dotProd As Single = 0.0F
                Dim weightOffset As Integer = j * dEmbd
                For k As Integer = 0 To dEmbd - 1
                    dotProd += Float16Converter.Fp16ToFp32(inp(inpOffset + k)) * Float16Converter.Fp16ToFp32(weight(weightOffset + k))
                Next
                output(j) = dotProd
            Next
        End Sub

        Public Shared Sub SiluInplace(inp As Array, nCtx As Integer, dEmbd As Integer, startPos As Integer, dtype As DType)
            If dtype = DType.Float16 Then
                Dim arr = CType(inp, UShort())
                For i As Integer = startPos To nCtx - 1
                    Dim offset As Integer = i * dEmbd
                    For j As Integer = 0 To dEmbd - 1
                        Dim x As Single = Float16Converter.Fp16ToFp32(arr(offset + j))
                        arr(offset + j) = Float16Converter.Fp32ToFp16(x / (1.0F + CSng(Math.Exp(-x))))
                    Next
                Next
            Else
                Dim arr = CType(inp, Single())
                For i As Integer = startPos To nCtx - 1
                    Dim offset As Integer = i * dEmbd
                    For j As Integer = 0 To dEmbd - 1
                        Dim x As Single = arr(offset + j)
                        arr(offset + j) = x / (1.0F + CSng(Math.Exp(-x)))
                    Next
                Next
            End If
        End Sub

        Public Shared Sub Matmul2D(inp0 As Array, inp1 As Array, output As Array, nCtx As Integer, dEmbd As Integer, nOut As Integer, startPos As Integer, dtype As DType)
            If dtype = DType.Float16 Then
                Matmul2DF16(CType(inp0, UShort()), CType(inp1, UShort()), CType(output, UShort()), nCtx, dEmbd, nOut, startPos)
            Else
                Matmul2DF32(CType(inp0, Single()), CType(inp1, Single()), CType(output, Single()), nCtx, dEmbd, nOut, startPos)
            End If
        End Sub

        Private Shared Sub Matmul2DF32(inp0 As Single(), inp1 As Single(), output As Single(), nCtx As Integer, dEmbd As Integer, nOut As Integer, startPos As Integer)
            For i As Integer = startPos To nCtx - 1
                Dim inp0Offset As Integer = i * dEmbd
                Dim outOffset As Integer = i * nOut
                For j As Integer = 0 To nOut - 1
                    Dim dotProd As Single = 0.0F
                    Dim inp1Offset As Integer = j * dEmbd
                    For k As Integer = 0 To dEmbd - 1
                        dotProd += inp0(inp0Offset + k) * inp1(inp1Offset + k)
                    Next
                    output(outOffset + j) = dotProd
                Next
            Next
        End Sub

        Private Shared Sub Matmul2DF16(inp0 As UShort(), inp1 As UShort(), output As UShort(), nCtx As Integer, dEmbd As Integer, nOut As Integer, startPos As Integer)
            For i As Integer = startPos To nCtx - 1
                Dim inp0Offset As Integer = i * dEmbd
                Dim outOffset As Integer = i * nOut
                For j As Integer = 0 To nOut - 1
                    Dim dotProd As Single = 0.0F
                    Dim inp1Offset As Integer = j * dEmbd
                    For k As Integer = 0 To dEmbd - 1
                        dotProd += Float16Converter.Fp16ToFp32(inp0(inp0Offset + k)) * Float16Converter.Fp16ToFp32(inp1(inp1Offset + k))
                    Next
                    output(outOffset + j) = Float16Converter.Fp32ToFp16(dotProd)
                Next
            Next
        End Sub

        Public Shared Sub QK(q As Array, k As Array, output As Array, nCtx As Integer, qHeads As Integer, kvHeads As Integer, dHead As Integer, scaler As Single, startPos As Integer, dtype As DType)
            If dtype = DType.Float16 Then
                QKF16(CType(q, UShort()), CType(k, UShort()), CType(output, UShort()), nCtx, qHeads, kvHeads, dHead, scaler, startPos)
            Else
                QKF32(CType(q, Single()), CType(k, Single()), CType(output, Single()), nCtx, qHeads, kvHeads, dHead, scaler, startPos)
            End If
        End Sub

        Private Shared Sub QKF32(q As Single(), k As Single(), output As Single(), nCtx As Integer, qHeads As Integer, kvHeads As Integer, dHead As Integer, scaler As Single, startPos As Integer)
            Dim kHeads As Integer = kvHeads
            Dim qGroupSize As Integer = qHeads \ kHeads
            For h As Integer = 0 To qHeads - 1
                Dim hk As Integer = h \ qGroupSize
                For i As Integer = startPos To nCtx - 1
                    For j As Integer = 0 To nCtx - 1
                        Dim dotProd As Single = 0.0F
                        Dim qOffset As Integer = h * dHead + i * qHeads * dHead
                        Dim kOffset As Integer = hk * dHead + j * kHeads * dHead
                        For kk As Integer = 0 To dHead - 1
                            dotProd += q(qOffset + kk) * k(kOffset + kk)
                        Next
                        output(h * nCtx * nCtx + i * nCtx + j) = dotProd * scaler
                    Next
                Next
            Next
        End Sub

        Private Shared Sub QKF16(q As UShort(), k As UShort(), output As UShort(), nCtx As Integer, qHeads As Integer, kvHeads As Integer, dHead As Integer, scaler As Single, startPos As Integer)
            Dim kHeads As Integer = kvHeads
            Dim qGroupSize As Integer = qHeads \ kHeads
            For h As Integer = 0 To qHeads - 1
                Dim hk As Integer = h \ qGroupSize
                For i As Integer = startPos To nCtx - 1
                    For j As Integer = 0 To nCtx - 1
                        Dim dotProd As Single = 0.0F
                        Dim qOffset As Integer = h * dHead + i * qHeads * dHead
                        Dim kOffset As Integer = hk * dHead + j * kHeads * dHead
                        For kk As Integer = 0 To dHead - 1
                            dotProd += Float16Converter.Fp16ToFp32(q(qOffset + kk)) * Float16Converter.Fp16ToFp32(k(kOffset + kk))
                        Next
                        output(h * nCtx * nCtx + i * nCtx + j) = Float16Converter.Fp32ToFp16(dotProd * scaler)
                    Next
                Next
            Next
        End Sub

        Public Shared Sub AttnMaskInplace(inp As Array, nHeads As Integer, nCtx As Integer, startPos As Integer, dtype As DType)
            Dim negInf As Single = Single.NegativeInfinity
            Dim negInf16 As UShort = Float16Converter.Fp32ToFp16(negInf)
            Dim isF16 = (dtype = DType.Float16)
            Dim arrF32 = If(isF16, Nothing, CType(inp, Single()))
            Dim arrF16 = If(isF16, CType(inp, UShort()), Nothing)

            For i As Integer = startPos To nHeads - 1
                For j As Integer = 0 To nCtx - 1
                    Dim startIdx As Integer = j + 1
                    Dim baseOffset As Integer = i * nCtx * nCtx + j * nCtx
                    For k As Integer = startIdx To nCtx - 1
                        If isF16 Then
                            arrF16(baseOffset + k) = negInf16
                        Else
                            arrF32(baseOffset + k) = negInf
                        End If
                    Next
                Next
            Next
        End Sub

        Public Shared Sub SoftmaxInplace(inp As Array, nHeads As Integer, nCtx As Integer, startPos As Integer, dtype As DType)
            If dtype = DType.Float16 Then
                SoftmaxInplaceF16(CType(inp, UShort()), nHeads, nCtx, startPos)
            Else
                SoftmaxInplaceF32(CType(inp, Single()), nHeads, nCtx, startPos)
            End If
        End Sub

        Private Shared Sub SoftmaxInplaceF32(inp As Single(), nHeads As Integer, nCtx As Integer, startPos As Integer)
            For h As Integer = 0 To nHeads - 1
                For i As Integer = startPos To nCtx - 1
                    Dim max As Single = Single.NegativeInfinity
                    Dim baseOffset As Integer = h * nCtx * nCtx + i * nCtx
                    For j As Integer = 0 To nCtx - 1
                        If inp(baseOffset + j) > max Then
                            max = inp(baseOffset + j)
                        End If
                    Next
                    Dim sumExp As Single = 0.0F
                    For j As Integer = 0 To nCtx - 1
                        Dim idx As Integer = baseOffset + j
                        inp(idx) = CSng(Math.Exp(inp(idx) - max))
                        sumExp += inp(idx)
                    Next
                    For j As Integer = 0 To nCtx - 1
                        inp(baseOffset + j) /= sumExp
                    Next
                Next
            Next
        End Sub

        Private Shared Sub SoftmaxInplaceF16(inp As UShort(), nHeads As Integer, nCtx As Integer, startPos As Integer)
            For h As Integer = 0 To nHeads - 1
                For i As Integer = startPos To nCtx - 1
                    Dim max As Single = Single.NegativeInfinity
                    Dim baseOffset As Integer = h * nCtx * nCtx + i * nCtx
                    For j As Integer = 0 To nCtx - 1
                        Dim val As Single = Float16Converter.Fp16ToFp32(inp(baseOffset + j))
                        If val > max Then
                            max = val
                        End If
                    Next
                    Dim sumExp As Single = 0.0F
                    For j As Integer = 0 To nCtx - 1
                        Dim idx As Integer = baseOffset + j
                        Dim res As Single = CSng(Math.Exp(Float16Converter.Fp16ToFp32(inp(idx)) - max))
                        inp(idx) = Float16Converter.Fp32ToFp16(res)
                        sumExp += res
                    Next
                    For j As Integer = 0 To nCtx - 1
                        Dim idx As Integer = baseOffset + j
                        inp(idx) = Float16Converter.Fp32ToFp16(Float16Converter.Fp16ToFp32(inp(idx)) / sumExp)
                    Next
                Next
            Next
        End Sub

        Public Shared Sub QKV(qk As Array, v As Array, output As Array, nCtx As Integer, qHeads As Integer, kvHeads As Integer, dHead As Integer, startPos As Integer, dtype As DType)
            If dtype = DType.Float16 Then
                QKVF16(CType(qk, UShort()), CType(v, UShort()), CType(output, UShort()), nCtx, qHeads, kvHeads, dHead, startPos)
            Else
                QKVF32(CType(qk, Single()), CType(v, Single()), CType(output, Single()), nCtx, qHeads, kvHeads, dHead, startPos)
            End If
        End Sub

        Private Shared Sub QKVF32(qk As Single(), v As Single(), output As Single(), nCtx As Integer, qHeads As Integer, kvHeads As Integer, dHead As Integer, startPos As Integer)
            Dim vHeads As Integer = kvHeads
            Dim qkGroupSize As Integer = qHeads \ vHeads
            For h As Integer = 0 To qHeads - 1
                Dim hv As Integer = h \ qkGroupSize
                For i As Integer = startPos To nCtx - 1
                    For j As Integer = 0 To dHead - 1
                        Dim dotProd As Single = 0.0F
                        Dim qkOffset As Integer = h * nCtx * nCtx + i * nCtx
                        For k As Integer = 0 To nCtx - 1
                            dotProd += qk(qkOffset + k) * v(hv * dHead + j + k * vHeads * dHead)
                        Next
                        output(i * qHeads * dHead + h * dHead + j) = dotProd
                    Next
                Next
            Next
        End Sub

        Private Shared Sub QKVF16(qk As UShort(), v As UShort(), output As UShort(), nCtx As Integer, qHeads As Integer, kvHeads As Integer, dHead As Integer, startPos As Integer)
            Dim vHeads As Integer = kvHeads
            Dim qkGroupSize As Integer = qHeads \ vHeads
            For h As Integer = 0 To qHeads - 1
                Dim hv As Integer = h \ qkGroupSize
                For i As Integer = startPos To nCtx - 1
                    For j As Integer = 0 To dHead - 1
                        Dim dotProd As Single = 0.0F
                        Dim qkOffset As Integer = h * nCtx * nCtx + i * nCtx
                        For k As Integer = 0 To nCtx - 1
                            dotProd += Float16Converter.Fp16ToFp32(qk(qkOffset + k)) * Float16Converter.Fp16ToFp32(v(hv * dHead + j + k * vHeads * dHead))
                        Next
                        output(i * qHeads * dHead + h * dHead + j) = Float16Converter.Fp32ToFp16(dotProd)
                    Next
                Next
            Next
        End Sub

        Public Shared Sub RotaryEmb(inp As Array, nCtx As Integer, nHeads As Integer, dHead As Integer, startPos As Integer, dtype As DType)
            If dtype = DType.Float16 Then
                RotaryEmbF16(CType(inp, UShort()), nCtx, nHeads, dHead, startPos)
            Else
                RotaryEmbF32(CType(inp, Single()), nCtx, nHeads, dHead, startPos)
            End If
        End Sub

        Private Shared Sub RotaryEmbF32(inp As Single(), nCtx As Integer, nHeads As Integer, dHead As Integer, startPos As Integer)
            For i As Integer = startPos To nCtx - 1
                For h As Integer = 0 To nHeads - 1
                    Dim inpVecOffset As Integer = i * nHeads * dHead + h * dHead
                    Dim dHalf As Integer = dHead \ 2
                    For j As Integer = 0 To dHalf - 1
                        Dim x0 As Single = inp(inpVecOffset + j)
                        Dim x1 As Single = inp(inpVecOffset + j + dHalf)

                        Dim m As Single = i
                        Dim inv_freq As Single = CalculateInvFreq(j, dHead)

                        Dim m_theta_i As Single = m * inv_freq
                        Dim o0 As Single = x0 * CSng(Math.Cos(m_theta_i)) - x1 * CSng(Math.Sin(m_theta_i))
                        Dim o1 As Single = x0 * CSng(Math.Sin(m_theta_i)) + x1 * CSng(Math.Cos(m_theta_i))
                        inp(inpVecOffset + j) = o0
                        inp(inpVecOffset + j + dHalf) = o1
                    Next
                Next
            Next
        End Sub

        Private Shared Sub RotaryEmbF16(inp As UShort(), nCtx As Integer, nHeads As Integer, dHead As Integer, startPos As Integer)
            For i As Integer = startPos To nCtx - 1
                For h As Integer = 0 To nHeads - 1
                    Dim inpVecOffset As Integer = i * nHeads * dHead + h * dHead
                    Dim dHalf As Integer = dHead \ 2
                    For j As Integer = 0 To dHalf - 1
                        Dim x0 As Single = Float16Converter.Fp16ToFp32(inp(inpVecOffset + j))
                        Dim x1 As Single = Float16Converter.Fp16ToFp32(inp(inpVecOffset + j + dHalf))

                        Dim m As Single = i
                        Dim inv_freq As Single = CalculateInvFreq(j, dHead)

                        Dim m_theta_i As Single = m * inv_freq
                        Dim o0 As Single = x0 * CSng(Math.Cos(m_theta_i)) - x1 * CSng(Math.Sin(m_theta_i))
                        Dim o1 As Single = x0 * CSng(Math.Sin(m_theta_i)) + x1 * CSng(Math.Cos(m_theta_i))
                        inp(inpVecOffset + j) = Float16Converter.Fp32ToFp16(o0)
                        inp(inpVecOffset + j + dHalf) = Float16Converter.Fp32ToFp16(o1)
                    Next
                Next
            Next
        End Sub

        Private Shared Function CalculateInvFreq(j As Integer, dHead As Integer) As Single
            Const base_theta As Single = 500000.0F
            Dim inv_freq As Single = CSng(Math.Pow(base_theta, -(2.0F * j / dHead)))

            Const factor As Single = 32.0F
            Const low_freq_factor As Single = 1.0F
            Const high_freq_factor As Single = 4.0F
            Const old_context_len As Single = 8192.0F
            Const low_freq_wavelen As Single = old_context_len / low_freq_factor
            Const high_freq_wavelen As Single = old_context_len / high_freq_factor
            Const pi As Single = CSng(Math.PI)
            Dim wavelen As Single = 2 * pi / inv_freq
            Dim inv_freq_llama As Single = inv_freq

            If wavelen > low_freq_wavelen Then
                inv_freq_llama = inv_freq / factor
            End If
            If wavelen <= low_freq_wavelen AndAlso wavelen >= high_freq_wavelen Then
                Dim smooth_factor As Single = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                inv_freq_llama = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
            End If
            Return inv_freq_llama
        End Function

        Public Shared Sub CopyTensors(src As Array, dest As Array, dtype As DType)
            Dim elementSize As Integer = If(dtype = DType.Float16, Marshal.SizeOf(GetType(UShort)), Marshal.SizeOf(GetType(Single)))
            Buffer.BlockCopy(src, 0, dest, 0, src.Length * elementSize)
        End Sub
    End Class

    ' --- Tokenizer Class ---
    Public Class Llama32Tokenizer
        Private ReadOnly _eotId As Integer = 128009
        Private ReadOnly _vocabSize As Integer
        Private ReadOnly _idToToken As TokenEntry()
        Private ReadOnly _tokenToId As TokenEntry() ' Sorted for binary search

        ' --- ИЗМЕНЕНИЕ НАЧАЛО: Добавлено скомпилированное регулярное выражение ---
        Private Shared ReadOnly TokenizerRegex As New Regex(
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}]+|[\p{N}]{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled)
        ' --- ИЗМЕНЕНИЕ КОНЕЦ ---

        Private Structure TokenEntry
            Public Token As String
            Public Id As Integer
        End Structure

        ' This comparer is used to sort the TokenEntry() array
        Private Class TokenEntryComparer
            Implements IComparer(Of TokenEntry)
            Public Function Compare(x As TokenEntry, y As TokenEntry) As Integer Implements IComparer(Of TokenEntry).Compare
                Return String.Compare(x.Token, y.Token, StringComparison.Ordinal)
            End Function
        End Class

        ' This comparer is used to search for a string in the TokenEntry() array
        ' It implements the non-generic IComparer, as required by Array.BinarySearch
        Private Class StringTokenComparer
            Implements System.Collections.IComparer
            Public Function Compare(x As Object, y As Object) As Integer Implements System.Collections.IComparer.Compare
                ' Determine which argument is a string and which is a structure
                If TypeOf x Is String AndAlso TypeOf y Is TokenEntry Then
                    Dim searchToken = CStr(x)
                    Dim arrayEntry = CType(y, TokenEntry)
                    Return String.Compare(searchToken, arrayEntry.Token, StringComparison.Ordinal)
                End If

                If TypeOf x Is TokenEntry AndAlso TypeOf y Is String Then
                    ' BinarySearch might swap the arguments, so handle this case as well.
                    ' The comparison result must be inverted.
                    Dim arrayEntry2 = CType(x, TokenEntry)
                    Dim searchToken2 = CStr(y)
                    Return -String.Compare(searchToken2, arrayEntry2.Token, StringComparison.Ordinal)
                End If

                ' If the types do not match expectations, it is an error in the calling logic.
                Throw New ArgumentException("Comparer can only compare a string and a TokenEntry.")
            End Function
        End Class

        Public ReadOnly Property EotId As Integer
            Get
                Return _eotId
            End Get
        End Property

        Public Sub New(vocabPath As String, nVocab As Integer)
            _vocabSize = nVocab
            _idToToken = New TokenEntry(nVocab - 1) {}
            _tokenToId = New TokenEntry(nVocab - 1) {}

            If Not File.Exists(vocabPath) Then
                Console.Error.WriteLine($"Failed to open vocab file: {vocabPath}")
                Environment.Exit(1)
            End If

            Using reader As New BinaryReader(File.OpenRead(vocabPath))
                For i As Integer = 0 To nVocab - 1
                    Dim length As Integer = reader.ReadInt32()
                    Dim bytes As Byte() = reader.ReadBytes(length)
                    Dim token As String = Encoding.UTF8.GetString(bytes)

                    _idToToken(i) = New TokenEntry With {.Token = token, .Id = i}
                    _tokenToId(i) = New TokenEntry With {.Token = token, .Id = i}
                Next
            End Using

            Array.Sort(_tokenToId, New TokenEntryComparer())
        End Sub

        Public Function Decode(tokenId As Integer) As String
            If tokenId < 0 OrElse tokenId >= _vocabSize Then Return Nothing
            Return _idToToken(tokenId).Token
        End Function

        Public Function Encode(text As String, tokens As Integer(), maxTokens As Integer) As Integer
            Dim encodePrefix = New Integer() {128000, 128006, 9125, 128007, 2675, 527, 264, 11190, 18328, 128009, 128006, 882, 128007}
            Dim encodeSuffix = New Integer() {128009, 128006, 78191, 128007}

            Dim tokenIdx As Integer = 0

            For Each t In encodePrefix
                If tokenIdx >= maxTokens Then Exit For
                tokens(tokenIdx) = t
                tokenIdx += 1
            Next

            ' --- ИЗМЕНЕНИЕ НАЧАЛО: Замена String.Split на Regex.Matches ---
            Dim words As New List(Of String)()
            For Each m As Match In TokenizerRegex.Matches(text)
                words.Add(m.Value)
            Next
            ' --- ИЗМЕНЕНИЕ КОНЕЦ ---

            ' Create an instance of the comparer once for reuse
            Dim comparer = New StringTokenComparer()

            For Each word In words
                If tokenIdx >= maxTokens - encodeSuffix.Length Then Exit For

                Dim i As Integer = 0
                Dim n As Integer = word.Length
                While i < n
                    If tokenIdx >= maxTokens - encodeSuffix.Length Then Exit While

                    Dim j As Integer = n
                    Dim found As Boolean = False
                    While j > i
                        Dim subStr As String = word.Substring(i, j - i)
                        ' Use the correct comparer
                        Dim searchResult As Integer = Array.BinarySearch(_tokenToId, subStr, comparer)

                        If searchResult >= 0 Then
                            tokens(tokenIdx) = _tokenToId(searchResult).Id
                            tokenIdx += 1
                            i = j
                            found = True
                            Exit While
                        End If
                        j -= 1
                    End While
                    If Not found Then
                        Dim subStr As String = word.Substring(i, 1)
                        ' Use the same comparer
                        Dim searchResult As Integer = Array.BinarySearch(_tokenToId, subStr, comparer)
                        If searchResult >= 0 Then
                            tokens(tokenIdx) = _tokenToId(searchResult).Id
                            tokenIdx += 1
                        Else
                            Console.Error.WriteLine($"Unknown token: '{subStr}'")
                        End If
                        i += 1
                    End If
                End While
            Next

            For Each t In encodeSuffix
                If tokenIdx >= maxTokens Then Exit For
                tokens(tokenIdx) = t
                tokenIdx += 1
            Next

            Return tokenIdx
        End Function
    End Class

    ' --- Model Structures ---
    Public Class Llama32Weights
        Public EmbTable As Array
        Public AttnNorm As Array() = New Array(Llama32Config.NumLayers - 1) {}
        Public QProj As Array() = New Array(Llama32Config.NumLayers - 1) {}
        Public KProj As Array() = New Array(Llama32Config.NumLayers - 1) {}
        Public VProj As Array() = New Array(Llama32Config.NumLayers - 1) {}
        Public OProj As Array() = New Array(Llama32Config.NumLayers - 1) {}
        Public MlpNorm As Array() = New Array(Llama32Config.NumLayers - 1) {}
        Public GateProj As Array() = New Array(Llama32Config.NumLayers - 1) {}
        Public UpProj As Array() = New Array(Llama32Config.NumLayers - 1) {}
        Public DownProj As Array() = New Array(Llama32Config.NumLayers - 1) {}
        Public OutNorm As Array
    End Class

    Public Class Llama32Acvs
        Public EmbAcv As Array
        Public AttnNormAcv As Array
        Public Res0Acv As Array
        Public Res1Acv As Array
        Public QProjAcv As Array() = New Array(Llama32Config.NumLayers - 1) {}
        Public KProjAcv As Array() = New Array(Llama32Config.NumLayers - 1) {}
        Public VProjAcv As Array() = New Array(Llama32Config.NumLayers - 1) {}
        Public OProjAcv As Array() = New Array(Llama32Config.NumLayers - 1) {}
        Public QkAcv As Array() = New Array(Llama32Config.NumLayers - 1) {}
        Public QkvAcv As Array() = New Array(Llama32Config.NumLayers - 1) {}
        Public MlpNormAcv As Array
        Public MlpGateAcv As Array() = New Array(Llama32Config.NumLayers - 1) {}
        Public MlpUpAcv As Array() = New Array(Llama32Config.NumLayers - 1) {}
        Public MlpDownAcv As Array() = New Array(Llama32Config.NumLayers - 1) {}
        Public OutNormAcv As Array
        Public LogitsAcv As Single()
    End Class

    ' --- Memory Management ---
    Private Function LlamaAlloc(n As Long, dtype As DType) As Array
        If dtype = DType.Float16 Then Return New UShort(CInt(n) - 1) {}
        Return New Single(CInt(n) - 1) {}
    End Function

    Private Function AllocLlama32Weights(dtype As DType) As Llama32Weights
        Dim w = New Llama32Weights()
        w.EmbTable = LlamaAlloc(CLng(Llama32Config.NVocab) * Llama32Config.DEmbd, dtype)
        For i As Integer = 0 To Llama32Config.NLayers - 1
            w.AttnNorm(i) = LlamaAlloc(Llama32Config.DEmbd, dtype)
            w.QProj(i) = LlamaAlloc(CLng(Llama32Config.NHeads) * Llama32Config.DHead * Llama32Config.DEmbd, dtype)
            w.KProj(i) = LlamaAlloc(CLng(Llama32Config.NKvHeads) * Llama32Config.DHead * Llama32Config.DEmbd, dtype)
            w.VProj(i) = LlamaAlloc(CLng(Llama32Config.NKvHeads) * Llama32Config.DHead * Llama32Config.DEmbd, dtype)
            w.OProj(i) = LlamaAlloc(CLng(Llama32Config.NHeads) * Llama32Config.DHead * Llama32Config.DEmbd, dtype)
            w.MlpNorm(i) = LlamaAlloc(Llama32Config.DEmbd, dtype)
            w.GateProj(i) = LlamaAlloc(CLng(Llama32Config.DMlp) * Llama32Config.DEmbd, dtype)
            w.UpProj(i) = LlamaAlloc(CLng(Llama32Config.DMlp) * Llama32Config.DEmbd, dtype)
            w.DownProj(i) = LlamaAlloc(CLng(Llama32Config.DMlp) * Llama32Config.DEmbd, dtype)
        Next
        w.OutNorm = LlamaAlloc(Llama32Config.DEmbd, dtype)
        Return w
    End Function

    Private Function AllocLlama32Acvs(dtype As DType, maxCtx As Integer) As Llama32Acvs
        Dim a = New Llama32Acvs()
        a.EmbAcv = LlamaAlloc(CLng(maxCtx) * Llama32Config.DEmbd, dtype)
        a.AttnNormAcv = LlamaAlloc(CLng(maxCtx) * Llama32Config.DEmbd, dtype)
        a.Res0Acv = LlamaAlloc(CLng(maxCtx) * Llama32Config.DEmbd, dtype)
        a.Res1Acv = LlamaAlloc(CLng(maxCtx) * Llama32Config.DEmbd, dtype)
        For i As Integer = 0 To Llama32Config.NLayers - 1
            a.QProjAcv(i) = LlamaAlloc(CLng(maxCtx) * Llama32Config.DEmbd, dtype)
            a.KProjAcv(i) = LlamaAlloc(CLng(maxCtx) * Llama32Config.DEmbd, dtype)
            a.VProjAcv(i) = LlamaAlloc(CLng(maxCtx) * Llama32Config.DEmbd, dtype)
            a.OProjAcv(i) = LlamaAlloc(CLng(maxCtx) * Llama32Config.DEmbd, dtype)
            a.QkAcv(i) = LlamaAlloc(CLng(Llama32Config.NHeads) * maxCtx * maxCtx, dtype)
            a.QkvAcv(i) = LlamaAlloc(CLng(maxCtx) * Llama32Config.NHeads * Llama32Config.DHead, dtype)
            a.MlpGateAcv(i) = LlamaAlloc(CLng(maxCtx) * Llama32Config.DMlp, dtype)
            a.MlpUpAcv(i) = LlamaAlloc(CLng(maxCtx) * Llama32Config.DMlp, dtype)
            a.MlpDownAcv(i) = LlamaAlloc(CLng(maxCtx) * Llama32Config.DEmbd, dtype)
        Next
        a.MlpNormAcv = LlamaAlloc(CLng(maxCtx) * Llama32Config.DEmbd, dtype)
        a.OutNormAcv = LlamaAlloc(CLng(maxCtx) * Llama32Config.DEmbd, dtype)
        a.LogitsAcv = CType(LlamaAlloc(Llama32Config.NVocab, DType.Float32), Single())
        Return a
    End Function

    Private Sub InitLlama32Weights(fpath As String, w As Llama32Weights, dtype As DType)
        If Not File.Exists(fpath) Then
            Console.Error.WriteLine($"Failed to open {fpath}.")
            Environment.Exit(1)
        End If

        Using reader As New BinaryReader(File.OpenRead(fpath))
            Const trueMagicNo As Long = &H663233616D616C6CL ' "llama32f"
            Dim magicNo As Long = reader.ReadInt64()
            Llama32Assert(magicNo = trueMagicNo, "Magic number mismatch.")

            Dim readBuffer As Action(Of Array) = Sub(arr As Array)
                                                    Dim elementSize As Integer = If(dtype = DType.Float16, Marshal.SizeOf(GetType(UShort)), Marshal.SizeOf(GetType(Single)))
                                                    Dim byteCount As Integer = arr.Length * elementSize
                                                    Dim bytesRead As Byte() = reader.ReadBytes(byteCount)
                                                    Llama32Assert(bytesRead.Length = byteCount, "Failed to read weights file.")
                                                    Buffer.BlockCopy(bytesRead, 0, arr, 0, byteCount)
                                                End Sub

            readBuffer(w.EmbTable)
            For i As Integer = 0 To Llama32Config.NLayers - 1
                readBuffer(w.AttnNorm(i))
                readBuffer(w.QProj(i))
                readBuffer(w.KProj(i))
                readBuffer(w.VProj(i))
                readBuffer(w.OProj(i))
                readBuffer(w.MlpNorm(i))
                readBuffer(w.GateProj(i))
                readBuffer(w.UpProj(i))
                readBuffer(w.DownProj(i))
            Next
            readBuffer(w.OutNorm)
        End Using
    End Sub

    ' --- Forward Pass ---
    Private Function Forward(tokens As Integer(), nCtx As Integer, w As Llama32Weights, a As Llama32Acvs, startPos As Integer, dtype As DType) As Single()
        Ops.Embed(tokens, w.EmbTable, a.EmbAcv, nCtx, Llama32Config.DEmbd, dtype)
        Dim nextLayerInp As Array = a.EmbAcv

        For i As Integer = 0 To Llama32Config.NLayers - 1
            Ops.CopyTensors(nextLayerInp, a.Res0Acv, dtype)
            Ops.RmsNorm(nextLayerInp, w.AttnNorm(i), a.AttnNormAcv, nCtx, Llama32Config.DEmbd, Llama32Config.RmsNormEps, dtype)

            Dim qDim As Integer = Llama32Config.NHeads * Llama32Config.DHead
            Dim kvDim As Integer = Llama32Config.NKvHeads * Llama32Config.DHead
            Ops.Matmul2D(a.AttnNormAcv, w.QProj(i), a.QProjAcv(i), nCtx, Llama32Config.DEmbd, qDim, startPos, dtype)
            Ops.Matmul2D(a.AttnNormAcv, w.KProj(i), a.KProjAcv(i), nCtx, Llama32Config.DEmbd, kvDim, startPos, dtype)
            Ops.Matmul2D(a.AttnNormAcv, w.VProj(i), a.VProjAcv(i), nCtx, Llama32Config.DEmbd, kvDim, startPos, dtype)
            Ops.RotaryEmb(a.QProjAcv(i), nCtx, Llama32Config.NHeads, Llama32Config.DHead, startPos, dtype)
            Ops.RotaryEmb(a.KProjAcv(i), nCtx, Llama32Config.NKvHeads, Llama32Config.DHead, startPos, dtype)

            Dim qkScaler As Single = 1.0F / CSng(Math.Sqrt(Llama32Config.DHead))
            Ops.QK(a.QProjAcv(i), a.KProjAcv(i), a.QkAcv(i), nCtx, Llama32Config.NHeads, Llama32Config.NKvHeads, Llama32Config.DHead, qkScaler, startPos, dtype)
            Ops.AttnMaskInplace(a.QkAcv(i), Llama32Config.NHeads, nCtx, startPos, dtype)
            Ops.SoftmaxInplace(a.QkAcv(i), Llama32Config.NHeads, nCtx, startPos, dtype)
            Ops.QKV(a.QkAcv(i), a.VProjAcv(i), a.QkvAcv(i), nCtx, Llama32Config.NHeads, Llama32Config.NKvHeads, Llama32Config.DHead, startPos, dtype)
            Ops.Matmul2D(a.QkvAcv(i), w.OProj(i), a.OProjAcv(i), nCtx, Llama32Config.DEmbd, Llama32Config.DEmbd, startPos, dtype)

            Ops.Residual(a.OProjAcv(i), a.Res0Acv, a.Res1Acv, nCtx, Llama32Config.DEmbd, dtype)

            Ops.RmsNorm(a.Res1Acv, w.MlpNorm(i), a.MlpNormAcv, nCtx, Llama32Config.DEmbd, Llama32Config.RmsNormEps, dtype)
            Ops.Matmul2D(a.MlpNormAcv, w.GateProj(i), a.MlpGateAcv(i), nCtx, Llama32Config.DEmbd, Llama32Config.DMlp, startPos, dtype)
            Ops.Matmul2D(a.MlpNormAcv, w.UpProj(i), a.MlpUpAcv(i), nCtx, Llama32Config.DEmbd, Llama32Config.DMlp, startPos, dtype)
            Ops.SiluInplace(a.MlpGateAcv(i), nCtx, Llama32Config.DMlp, startPos, dtype)
            Ops.MulInplace(a.MlpGateAcv(i), a.MlpUpAcv(i), nCtx, Llama32Config.DMlp, startPos, dtype)
            Ops.Matmul2D(a.MlpGateAcv(i), w.DownProj(i), a.MlpDownAcv(i), nCtx, Llama32Config.DMlp, Llama32Config.DEmbd, startPos, dtype)
            Ops.Residual(a.Res1Acv, a.MlpDownAcv(i), a.Res1Acv, nCtx, Llama32Config.DEmbd, dtype)

            nextLayerInp = a.Res1Acv
        Next

        Ops.RmsNorm(nextLayerInp, w.OutNorm, a.OutNormAcv, nCtx, Llama32Config.DEmbd, Llama32Config.RmsNormEps, dtype)
        Ops.LmHeadProj(a.OutNormAcv, w.EmbTable, a.LogitsAcv, Llama32Config.NVocab, nCtx, Llama32Config.DEmbd, dtype)

        Return a.LogitsAcv
    End Function

    ' --- Sampling ---
    Private Structure LogitProb
        Public Prob As Single
        Public Index As Integer
    End Structure

    Private Function CompareLogitProb(a As LogitProb, b As LogitProb) As Integer
        Return b.Prob.CompareTo(a.Prob)
    End Function

    Private Function TopKSample(prompt As String, w As Llama32Weights, a As Llama32Acvs, tokenizer As Llama32Tokenizer, dtype As DType, maxCtx As Integer, temp As Single, topK As Integer) As Integer
        Dim tokens As Integer() = New Integer(8191) {}
        Dim nTokens As Integer = tokenizer.Encode(prompt, tokens, maxCtx)
        If nTokens >= maxCtx Then
            Console.Error.WriteLine($"Prompt too large: {nTokens} for max context {maxCtx}")
            Return 0
        End If

        Dim logitsSize As Integer = Llama32Config.NVocab
        Dim logitProbs As LogitProb() = New LogitProb(logitsSize - 1) {}
        Dim rand = New Random()
        Dim eotToken As Integer = tokenizer.EotId
        Dim nPredTokens As Integer = maxCtx - nTokens

        For i As Integer = 0 To nPredTokens - 1
            Dim startPos As Integer = If(i = 0, 0, nTokens - 1)
            Dim logits As Single() = Forward(tokens, nTokens, w, a, startPos, dtype)

            For j As Integer = 0 To logitsSize - 1
                logitProbs(j).Prob = logits(j) / temp
                logitProbs(j).Index = j
            Next

            Array.Sort(logitProbs, AddressOf CompareLogitProb)

            Dim sumExp As Single = 0.0F
            For j As Integer = 0 To topK - 1
                logitProbs(j).Prob = CSng(Math.Exp(logitProbs(j).Prob))
                sumExp += logitProbs(j).Prob
            Next
            For j As Integer = 0 To topK - 1
                logitProbs(j).Prob /= sumExp
            Next

            Dim r As Single = CSng(rand.NextDouble())
            Dim cumsum As Single = 0.0F
            Dim predToken As Integer = -1
            For j As Integer = 0 To topK - 1
                cumsum += logitProbs(j).Prob
                If r < cumsum Then
                    predToken = logitProbs(j).Index
                    Exit For
                End If
            Next
            If predToken = -1 Then predToken = logitProbs(topK - 1).Index

            If predToken = eotToken Then Exit For

            Console.Write(tokenizer.Decode(predToken))

            tokens(nTokens) = predToken
            nTokens += 1
        Next
        Console.WriteLine()
        Return nTokens
    End Function

    ' --- Main Function ---
    Private Const UsageMessage As String =
        "USAGE:" & vbCrLf &
        "./llama32 [options] -p PROMPT  for a single prompt or" & vbCrLf &
        "./llama32 [options] for a chat interface." & vbCrLf & vbCrLf &
        "Optional args:" & vbCrLf &
        "-f16 :     Use float-16 model (2.3GB). [default]" & vbCrLf &
        "--npred N : Max context size. Minimum is 128 and max is 8192 [default=512]. Higher values consume more memory." & vbCrLf

    Public Sub RunApplication(args As String())
        Dim modelPath As String = "models/llama32-1B.fp16.bin"
        Dim modelDtype As DType = DType.Float16
        Dim maxCtx As Integer = 512
        Dim prompt As String = Nothing

        Try
            Dim i As Integer = 0
            While i < args.Length
                Select Case args(i)
                    Case "--help", "-h"
                        Console.WriteLine(UsageMessage)
                        Return
                    Case "-f16"
                        ' Continue
                    Case "-p"
                        If i + 1 < args.Length Then
                            i += 1
                            prompt = args(i)
                        Else
                            Throw New ArgumentException("Prompt not provided.")
                        End If
                    Case "--npred"
                        i += 1
                        If i >= args.Length Then Throw New ArgumentException("npred value missing.")
                        Dim npred As Integer = Integer.Parse(args(i))
                        If npred < 128 OrElse npred > 8192 Then Throw New ArgumentException("npred must be between 128 and 8192.")
                        maxCtx = npred
                    Case Else
                        Throw New ArgumentException($"Unknown argument: {args(i)}")
                End Select
                i += 1
            End While
        Catch ex As Exception
            Console.Error.WriteLine(ex.Message)
            Console.Error.WriteLine(UsageMessage)
            Environment.Exit(-1)
        End Try

        Try
            Dim processInfo As New ProcessStartInfo With {
                .FileName = "python3",
                .Arguments = "model_dl.py",
                .UseShellExecute = False,
                .RedirectStandardOutput = True,
                .RedirectStandardError = True
            }
            If RuntimeInformation.IsOSPlatform(OSPlatform.Windows) Then
                processInfo.FileName = "python"
            End If

            Using process As Process = Process.Start(processInfo)
                process.WaitForExit()
                If process.ExitCode <> 0 Then
                    Console.Error.WriteLine("Failed to download model.")
                    Console.Error.WriteLine(process.StandardError.ReadToEnd())
                    Environment.Exit(-1)
                End If
            End Using
        Catch ex As Exception
            Console.Error.WriteLine("Failed to run model download script. Make sure Python is installed and in your PATH.")
            Console.Error.WriteLine(ex.Message)
            Environment.Exit(-1)
        End Try

        ' Initialize weights, activations, and tokenizer
        Dim w = AllocLlama32Weights(modelDtype)
        Dim a = AllocLlama32Acvs(modelDtype, maxCtx)
        InitLlama32Weights(modelPath, w, modelDtype)

        Dim vocabTokSize As Integer = 128000
        Dim tokenizer = New Llama32Tokenizer("tokenizer.bin", vocabTokSize)

        If String.IsNullOrEmpty(prompt) Then
            Console.WriteLine("Chat interface. Write your prompt and press enter to submit. Enter 'q' or Ctrl+C to quit.")
            While True
                Console.Write(vbCrLf & vbCrLf & "[You]: ")
                Dim chatPrompt As String = Console.ReadLine()
                If chatPrompt Is Nothing OrElse chatPrompt = "q" Then Exit While

                Console.Write(vbCrLf & vbCrLf & "[LLAMA-1B]: " & vbCrLf)
                TopKSample(chatPrompt, w, a, tokenizer, modelDtype, maxCtx, 0.9F, 40)
            End While
        Else
            Console.Write($"\n[PROMPT]:\n{prompt}\n\n[LLAMA-1B]: ")
            TopKSample(prompt, w, a, tokenizer, modelDtype, maxCtx, 0.9F, 40)
        End If
    End Sub

End Module