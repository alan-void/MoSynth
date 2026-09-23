using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEditor.UIElements;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.UIElements;

public class BoneSelectorWindow : EditorWindow
{
    private BoneNode _selectedNode;

    private VisualElement _canvas;
    private VisualElement _inspectorPanel;

    private SkeletonUIManager _uiManager;
    private ObjectField _uiManagerField;

    public static void ShowWindow(SkeletonUIManager uiManager)
    {
        var wnd = CreateWindow<BoneSelectorWindow>();
        wnd.titleContent = new GUIContent("Bone Selector");
        wnd._uiManager = uiManager;
        wnd._uiManagerField.SetValueWithoutNotify(uiManager);
        wnd.RefreshCanvas();
        wnd.RefreshInspector();
    }

    private readonly Dictionary<string, Button> _boneButtons = new Dictionary<string, Button>();

    public void CreateGUI()
    {
        var root = rootVisualElement;
        root.style.flexDirection = FlexDirection.Column;

        // Toolbar
        var toolbar = new Toolbar();
        root.Add(toolbar);

        _uiManagerField = new ObjectField("Skeleton UI Manager")
        {
            objectType = typeof(SkeletonUIManager),
            allowSceneObjects = true,
            value = _uiManager
        };
        _uiManagerField.RegisterValueChangedCallback(evt =>
        {
            _uiManager = evt.newValue as SkeletonUIManager;
            _selectedNode = null;
            RefreshCanvas();
            RefreshInspector();
        });
        toolbar.Add(_uiManagerField);
        
        // Main Area (Canvas + Inspector)
        var mainContainer = new VisualElement { style = { flexDirection = FlexDirection.Row, flexGrow = 1 } };
        root.Add(mainContainer);

        _canvas = new VisualElement
        {
            style =
            {
                flexGrow = 1,
                backgroundColor = new Color(0.15f, 0.15f, 0.15f, 1f),
                overflow = Overflow.Hidden,
                position = Position.Relative
            }
        };
        _canvas.AddManipulator(new ContextualMenuManipulator(evt =>
        {
            if (_uiManager == null)
            {
                evt.menu.AppendAction("Add Bone", null, DropdownMenuAction.Status.Disabled);
                return;
            }
            evt.menu.AppendAction("Add Bone", _ => AddBone());
        }));
        mainContainer.Add(_canvas);

        _inspectorPanel = new VisualElement
        {
            style =
            {
                width = 250,
                backgroundColor = new Color(0.2f, 0.2f, 0.2f, 1f),
                borderLeftWidth = 1,
                borderLeftColor = Color.black,
                paddingBottom = 10, paddingTop = 10, paddingLeft = 10, paddingRight = 10,
            }
        };
        mainContainer.Add(_inspectorPanel);
    }

    private void AddBone()
    {
        Assert.IsTrue(_uiManager != null, "SkeletonUIManager is not assigned.");

        var newNode = new BoneNode { id = Guid.NewGuid().ToString() };
        _uiManager.boneNodes.Add(newNode);
        EditorUtility.SetDirty(_uiManager);
        RefreshCanvas();
    }

    private void RefreshCanvas()
    {
        _canvas.Clear();
        _boneButtons.Clear();

        if (_uiManager == null)
        {
            _canvas.Add(new Label("Assign a Skeleton UI Manager to display bones."));
            return;
        }

        foreach (var node in _uiManager.boneNodes)
        {
            var btn = new Button
            {
                text = node.displayName,
                style =
                {
                    position = Position.Absolute,
                    left = node.rect.x,
                    top = node.rect.y,
                    width = node.rect.width,
                    height = node.rect.height
                }
            };

            btn.style.backgroundColor = new Color(0.3f, 0.5f, 0.8f);
            btn.AddManipulator(new NodeDragManipulator(btn, node, this));
            btn.clickable.clicked += () => SelectNodeForEdit(node);

            if (!string.IsNullOrEmpty(node.id))
            {
                _boneButtons[node.id] = btn;
            }
            _canvas.Add(btn);
        }
    }

    private void SelectNodeForEdit(BoneNode node)
    {
        _selectedNode = node;
        RefreshInspector();
    }

    private void RefreshInspector()
    {
        _inspectorPanel.Clear();

        if (_uiManager == null)
        {
            _inspectorPanel.Add(new Label("Assign a Skeleton UI Manager to edit bones."));
            return;
        }

        if (_selectedNode == null)
        {
            _inspectorPanel.Add(new Label("Select a bone to edit..."));
            return;
        }

        var nameField = new TextField("Label") { value = _selectedNode.displayName };
        nameField.RegisterValueChangedCallback(evt =>
        {
            _selectedNode.displayName = evt.newValue;
            SaveProfile();
            if (!string.IsNullOrEmpty(_selectedNode.id) && _boneButtons.TryGetValue(_selectedNode.id, out var button))
            {
                button.text = _selectedNode.displayName;
            }
        });
        _inspectorPanel.Add(nameField);

        var targetField = new ObjectField("Target Transform") { objectType = typeof(Transform), value = _selectedNode.boneTransform, allowSceneObjects = true };
        targetField.RegisterValueChangedCallback(evt => { _selectedNode.boneTransform = evt.newValue as Transform; SaveProfile(); });
        _inspectorPanel.Add(targetField);

        var selectInHierarchyBtn = new Button(() =>
            {
                if (_selectedNode?.boneTransform != null)
                {
                    Selection.activeGameObject = _selectedNode.boneTransform.gameObject;
                }
            })
            { text = "Select In Hierarchy" };
        selectInHierarchyBtn.SetEnabled(_selectedNode.boneTransform != null);
        _inspectorPanel.Add(selectInHierarchyBtn);

        var widthField = new FloatField("Width") { value = _selectedNode.rect.width };
        widthField.RegisterValueChangedCallback(evt =>
        {
            _selectedNode.rect.width = evt.newValue;
            SaveProfile();
            if (!string.IsNullOrEmpty(_selectedNode.id) && _boneButtons.TryGetValue(_selectedNode.id, out var button))
            {
                button.style.width = _selectedNode.rect.width;
            }
        });
        _inspectorPanel.Add(widthField);

        var heightField = new FloatField("Height") { value = _selectedNode.rect.height };
        heightField.RegisterValueChangedCallback(evt =>
        {
            _selectedNode.rect.height = evt.newValue;
            SaveProfile();
            if (!string.IsNullOrEmpty(_selectedNode.id) && _boneButtons.TryGetValue(_selectedNode.id, out var button))
            {
                button.style.height = _selectedNode.rect.height;
            }
        });
        _inspectorPanel.Add(heightField);

        var deleteBtn = new Button(() =>
            {
                _uiManager.boneNodes.Remove(_selectedNode);
                _selectedNode = null;
                SaveAndRefresh();
            })
            { text = "Delete Node", style = { marginTop = 20, backgroundColor = new Color(0.6f, 0.2f, 0.2f) } };
        _inspectorPanel.Add(deleteBtn);
    }

    public void SaveProfile()
    {
        EditorUtility.SetDirty(_uiManager);
        AssetDatabase.SaveAssetIfDirty(_uiManager);
    }

    private void SaveAndRefresh()
    {
        SaveProfile();
        RefreshCanvas();
    }
}

public class NodeDragManipulator : PointerManipulator
{
    private bool _isDragging;
    private Vector3 _startMousePosition;
    private Vector2 _startElementPosition;
    private readonly VisualElement _targetElement;
    private readonly BoneNode _nodeData;
    private readonly BoneSelectorWindow _window;

    public NodeDragManipulator(VisualElement target, BoneNode data, BoneSelectorWindow wnd)
    {
        _targetElement = target;
        _nodeData = data;
        _window = wnd;
    }

    protected override void RegisterCallbacksOnTarget()
    {
        target.RegisterCallback<PointerDownEvent>(OnPointerDown);
        target.RegisterCallback<PointerMoveEvent>(OnPointerMove);
        target.RegisterCallback<PointerUpEvent>(OnPointerUp);
        target.RegisterCallback<PointerCaptureOutEvent>(OnPointerCaptureOut);
    }

    protected override void UnregisterCallbacksFromTarget()
    {
        target.UnregisterCallback<PointerDownEvent>(OnPointerDown);
        target.UnregisterCallback<PointerMoveEvent>(OnPointerMove);
        target.UnregisterCallback<PointerUpEvent>(OnPointerUp);
        target.UnregisterCallback<PointerCaptureOutEvent>(OnPointerCaptureOut);
    }

    private void OnPointerDown(PointerDownEvent evt)
    {
        _isDragging = true;
        _startMousePosition = evt.position;
        _startElementPosition = new Vector2(_targetElement.layout.x, _targetElement.layout.y);
        target.CapturePointer(evt.pointerId);
    }

    private void OnPointerMove(PointerMoveEvent evt)
    {
        if (!_isDragging || !target.HasPointerCapture(evt.pointerId)) return;

        Vector2 diff = evt.position - _startMousePosition;
        _targetElement.style.left = _startElementPosition.x + diff.x;
        _targetElement.style.top = _startElementPosition.y + diff.y;
    }

    private void OnPointerUp(PointerUpEvent evt)
    {
        if (!_isDragging || !target.HasPointerCapture(evt.pointerId)) return;
        
        _isDragging = false;
        target.ReleasePointer(evt.pointerId);

        _nodeData.rect.x = _targetElement.layout.x;
        _nodeData.rect.y = _targetElement.layout.y;
        _window.SaveProfile();
    }

    private void OnPointerCaptureOut(PointerCaptureOutEvent evt)
    {
        _isDragging = false;
    }
}