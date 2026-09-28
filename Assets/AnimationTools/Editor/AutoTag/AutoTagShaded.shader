// Self-lit so a render does not depend on which lights or ambient a preview scene gets under the
// active pipeline; carries no LightMode tag, so URP draws it as SRPDefaultUnlit.
Shader "Hidden/MoSynth/AutoTagShaded"
{
    Properties
    {
        _Color ("Color", Color) = (1, 1, 1, 1)
        _CheckerColor ("Checker Color", Color) = (0.5, 0.5, 0.5, 1)
        _CheckerSize ("Checker Size (0 = solid)", Float) = 0
    }

    SubShader
    {
        Tags { "RenderType" = "Opaque" "Queue" = "Geometry" }

        Pass
        {
            Cull Off
            ZWrite On
            ZTest LEqual

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            float4 _Color;
            float4 _CheckerColor;
            float _CheckerSize;

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 worldNormal : TEXCOORD0;
                float3 worldPos : TEXCOORD1;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.worldNormal = UnityObjectToWorldNormal(v.normal);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                float3 albedo = _Color.rgb;
                if (_CheckerSize > 0)
                {
                    float2 uv = i.worldPos.xz / _CheckerSize;
                    float2 cell = floor(uv);
                    albedo = frac((cell.x + cell.y) * 0.5) < 0.25 ? _Color.rgb : _CheckerColor.rgb;

                    // Fades to the mean where squares shrink below a pixel or two, which would moire.
                    float2 footprint = fwidth(uv);
                    float fade = saturate(max(footprint.x, footprint.y) * 2 - 0.5);
                    albedo = lerp(albedo, (_Color.rgb + _CheckerColor.rgb) * 0.5, fade);
                }

                // Key light from above and behind the camera, so the lit side always faces the viewer.
                float3 toCamera = normalize(_WorldSpaceCameraPos - i.worldPos);
                float3 key = normalize(toCamera + float3(0, 1.2, 0));
                float3 n = normalize(i.worldNormal);
                if (dot(n, toCamera) < 0) n = -n;

                float shade = 0.35 + 0.65 * saturate(dot(n, key));
                return float4(albedo * shade, 1);
            }
            ENDCG
        }
    }
}
