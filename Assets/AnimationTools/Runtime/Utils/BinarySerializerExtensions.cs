using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using System.IO;

namespace AnimationTools
{
    /// <summary>
    /// Reads and writes the Unity.Mathematics types <see cref="BinaryWriter"/> has no overloads for.
    /// </summary>
    /// <remarks>
    /// Components in order, little-endian float32, no length prefix or padding — arrays are just
    /// their elements back to back, which is why the Read*Array methods need a length. That plain
    /// layout is what lets the Python side read these with one <c>numpy.fromfile</c> and a reshape.
    /// Quaternions are xyzw throughout.
    /// </remarks>
    public class BinarySerializerExtensions
    {
        public static void WriteFloat2(BinaryWriter writer, float2 value)
        {
            writer.Write(value.x);
            writer.Write(value.y);
        }
        public static void WriteFloat3(BinaryWriter writer, float3 value)
        {
            writer.Write(value.x);
            writer.Write(value.y);
            writer.Write(value.z);
        }
        public static void Write(BinaryWriter writer, Vector3 value)
        {
            writer.Write(value.x);
            writer.Write(value.y);
            writer.Write(value.z);
        }
        public static void WriteFloat4(BinaryWriter writer, float4 value)
        {
            writer.Write(value.x);
            writer.Write(value.y);
            writer.Write(value.z);
            writer.Write(value.w);
        }
        public static void WriteQuaternion(BinaryWriter writer, quaternion value)
        {
            writer.Write(value.value.x);
            writer.Write(value.value.y);
            writer.Write(value.value.z);
            writer.Write(value.value.w);
        }
        public static void WriteFloat3Array(BinaryWriter writer, float3[] array)
        {
            foreach (float3 value in array)
            {
                WriteFloat3(writer, value);
            }
        }

        public static void WriteFloat3Slice(BinaryWriter writer, NativeSlice<float3> slice)
        {
            for (var i = 0; i < slice.Length; i++)
            {
                WriteFloat3(writer, slice[i]);
            }
        }

        public static void WriteQuaternionSlice(BinaryWriter writer, NativeSlice<quaternion> slice)
        {
            for (var i = 0; i < slice.Length; i++)
            {
                WriteQuaternion(writer, slice[i]);
            }
        }
        
        public static void Write(BinaryWriter writer, Vector3[] array)
        {
            foreach (var value in array)
            {
                Write(writer, value);
            }
        }
        
        
        public static void WriteQuaternionArray(BinaryWriter writer, quaternion[] array)
        {
            foreach (quaternion value in array)
            {
                WriteQuaternion(writer, value);
            }
        }
        public static void WriteQuaternionArray(BinaryWriter writer, Quaternion[] array)
        {
            foreach (var value in array)
            {
                WriteQuaternion(writer, value);
            }
        }

        public static float2 ReadFloat2(BinaryReader reader)
        {
            return new float2(reader.ReadSingle(), reader.ReadSingle());
        }
        public static float3 ReadFloat3(BinaryReader reader)
        {
            return new float3(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
        }
        public static float4 ReadFloat4(BinaryReader reader)
        {
            return new float4(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
        }
        public static quaternion ReadQuaternion(BinaryReader reader)
        {
            return new quaternion(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());
        }
        public static float3[] ReadFloat3Array(BinaryReader reader, uint length)
        {
            float3[] array = new float3[length];
            for (int i = 0; i < length; i++)
            {
                array[i] = ReadFloat3(reader);
            }
            return array;
        }
        public static quaternion[] ReadQuaternionArray(BinaryReader reader, uint length)
        {
            quaternion[] array = new quaternion[length];
            for (int i = 0; i < length; i++)
            {
                array[i] = ReadQuaternion(reader);
            }
            return array;
        }
    }
}